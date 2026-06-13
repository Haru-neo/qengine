// Continuous batching scheduler for the qwen-engine inference server.
//
// Goals
//   - Accept N concurrent HTTP clients without serializing them at the socket
//     layer.
//   - Give each client its own KV+GDN slot so per-request state survives
//     overlapping requests, and so per-slot prefix caches work.
//   - Provide a stable extension point for real batched-forward fusion
//     (Phase B). The current implementation serializes forward execution
//     behind a single GPU mutex (round-robin); replacing the worker loop with
//     a batched scheduler is a drop-in change that does not affect the public
//     API of this header.
//
// Design
//   - SlotManager owns the [0, N) slot ids and hands them out FCFS. Allocate
//     blocks until a slot frees, so the HTTP layer can issue requests without
//     bookkeeping.
//   - Sequence is the per-request state: prompt + sampling + completion
//     callbacks + the slot id once allocated.
//   - GenScheduler runs a fixed pool of worker threads. Each worker pops a
//     pending Sequence, allocates a slot, calls user-supplied `run_fn(seq,
//     slot)` (which is generate_impl wrapped to thread `slot`), then releases
//     the slot. A `forward_mutex` is exposed so callers can serialize GPU
//     execution while still letting multiple host threads progress; Phase B
//     will replace the per-sequence run_fn with a batched scheduler that
//     does not need the global mutex.
//
// Thread-safety notes
//   - The forward_mutex is provided by the scheduler but acquired/released
//     inside run_fn. This lets run_fn drop the lock around long
//     non-GPU operations (token decoding, callback invocation) so other
//     workers can advance.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "json_grammar.h"

namespace qwen_engine {

struct Sequence {
    // Inputs
    std::vector<int> prompt_ids;
    int  max_tokens          = 0;
    int  cached_prompt_tokens = 0;
    // Optional response-format grammar (JSON-constrained decoding). When set,
    // run_fn must route this request through the single-slot legacy path so
    // the per-token sampler can apply the grammar mask. Continuous-batching
    // multi-slot mode does its own sampling that doesn't currently honor
    // grammars; allowing both would silently drop the constraint.
    std::shared_ptr<JsonGrammar> grammar;

    // Output sinks (run_fn calls these as the request progresses).
    //   on_token : called once per emitted token. Used by the SSE streaming
    //              wrapper; no-op for non-streaming requests.
    //   on_done  : called exactly once after run_fn finishes. Carries the
    //              final assembled text and total completion token count.
    std::function<void(int /*token_id*/)>                    on_token;
    std::function<void(std::string /*final_text*/, int /*completion_tokens*/)> on_done;

    // If >=0 and < num_slots, the scheduler waits for this specific slot
    // to be free instead of taking the first available one. Lets clients
    // pin a "session" to a slot for prefix-cache reuse across requests.
    int requested_slot = -1;

    // Filled by the scheduler once a slot is allocated.
    int slot_id = -1;
    // Set true when run_fn finishes — observers can poll without the
    // scheduler's mutex.
    std::atomic<bool> finished{false};
    // Set true when the client has disconnected (e.g. SSE send returned
    // EPIPE). generate_impl + the batched gen loop poll this in their
    // per-token loops and break early so we don't waste GPU cycles
    // generating tokens nobody will receive.
    std::atomic<bool> cancelled{false};
    // Repetition-loop cut signal: 0 = none, >0 = the detected cycle period.
    // Set by the rep-loop detector when it stops this generation. The HTTP
    // layer uses it to (a) auto-resume the request with the looped tail
    // collapsed to 2 cycles (REP_LOOP_RESUME, breaks the deterministic
    // greedy loop by changing the context), and (b) mark the response
    // finish_reason "length" when resumes are exhausted — agent clients must
    // be able to tell a truncated answer from a complete one.
    std::atomic<int> rep_cut{0};
};

class SlotManager {
public:
    explicit SlotManager(int num_slots) : free_(num_slots, true) {}

    int num_slots() const { return (int)free_.size(); }

    // Allocate a free slot. Blocks until one is available. Thread-safe.
    int allocate() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]() {
            for (size_t i = 0; i < free_.size(); ++i)
                if (free_[i]) return true;
            return false;
        });
        for (size_t i = 0; i < free_.size(); ++i) {
            if (free_[i]) { free_[i] = false; return (int)i; }
        }
        return -1;  // unreachable
    }

    // Wait for a specific slot to become free, then claim it. Used for
    // client-pinned sessions. Returns -1 if the slot index is out of range.
    int allocate_specific(int slot) {
        if (slot < 0 || slot >= (int)free_.size()) return -1;
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]() { return free_[slot]; });
        free_[slot] = false;
        return slot;
    }

    // Return a slot to the pool.
    void release(int slot) {
        if (slot < 0 || slot >= (int)free_.size()) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            free_[slot] = true;
        }
        // notify_all (not notify_one) because waiters have heterogeneous
        // predicates: `allocate()` waits on "any slot free", `allocate_specific(N)`
        // waits on "slot N free". A notify_one can wake the wrong waiter,
        // starving a specific-slot waiter while a generic waiter consumes
        // the freed slot.
        cv_.notify_all();
    }

private:
    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<bool>       free_;
};

// Thread pool that runs Sequence-bound work. The actual generation (prefill +
// per-token loop) is supplied by `run_fn` which is closed over the model and
// helper buffers from main.cu's serve_qwen.
class GenScheduler {
public:
    using RunFn = std::function<void(Sequence& seq, int slot)>;

    GenScheduler(int num_slots, RunFn run_fn, int num_workers = 0)
        : slots_(num_slots),
          run_fn_(std::move(run_fn)),
          stop_(false)
    {
        if (num_workers <= 0) num_workers = num_slots;
        workers_.reserve(num_workers);
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~GenScheduler() {
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            stop_ = true;
        }
        queue_cv_.notify_all();
        for (auto& th : workers_) if (th.joinable()) th.join();
    }

    // Submit a sequence for execution. Non-blocking; the caller usually owns
    // the Sequence via shared_ptr and waits on `finished` or `on_done`.
    // Returns false if the queue is at its configured cap (QWEN_MAX_QUEUE
    // env), in which case the caller should reject the request (HTTP 503).
    bool submit(std::shared_ptr<Sequence> seq) {
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            if (max_queue_ > 0 && (int)queue_.size() >= max_queue_) return false;
            queue_.emplace_back(std::move(seq));
        }
        queue_cv_.notify_one();
        return true;
    }

    // Configure max queued (pending, not in-flight) sequences. 0 = unbounded.
    // Use to apply backpressure under bursty load instead of letting the
    // queue grow without bound.
    void set_max_queue(int max_queue) { max_queue_ = max_queue; }
    int  queued_count() {
        std::lock_guard<std::mutex> lk(queue_mu_);
        return (int)queue_.size();
    }

    // Speculative-prefill prefetch (P3): return a COPY of the next pending
    // sequence's prompt_ids IF it is at least `min_len` tokens and not already
    // cancelled, else empty. Used to kick off the kvgen sidecar for the next
    // long prompt while the current one decodes. Non-destructive (does not pop)
    // — the sequence is still run normally; the prefetched KV is matched back by
    // exact prompt hash, so a peek that never gets used is harmless.
    std::vector<int> peek_next_prompt(int min_len) {
        std::lock_guard<std::mutex> lk(queue_mu_);
        for (const auto& s : queue_) {
            if (s->cancelled.load(std::memory_order_acquire)) continue;
            if ((int)s->prompt_ids.size() >= min_len) return s->prompt_ids;
            return {};  // only consider the head; deeper entries may reorder
        }
        return {};
    }

    // Mutex held by run_fn around GPU forward execution. Phase A serializes
    // every forward call; Phase B replaces this with a batched scheduler so
    // forward is naturally batched, not mutex-protected.
    std::mutex& forward_mutex() { return forward_mu_; }

    int num_slots() const { return slots_.num_slots(); }

private:
    void worker_loop() {
        while (true) {
            std::shared_ptr<Sequence> seq;
            {
                std::unique_lock<std::mutex> lk(queue_mu_);
                queue_cv_.wait(lk, [&]() { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                seq = std::move(queue_.front());
                queue_.pop_front();
            }
            // Drop already-cancelled sequences (client gave up while
            // waiting in the queue) without burning a slot on them.
            if (seq->cancelled.load(std::memory_order_acquire)) {
                seq->finished.store(true, std::memory_order_release);
                continue;
            }
            int slot;
            if (seq->requested_slot >= 0 && seq->requested_slot < slots_.num_slots()) {
                slot = slots_.allocate_specific(seq->requested_slot);
                if (slot < 0) slot = slots_.allocate();
            } else {
                slot = slots_.allocate();
            }
            seq->slot_id = slot;
            try {
                run_fn_(*seq, slot);
            } catch (const std::exception& e) {
                fprintf(stderr, "[scheduler] run_fn threw: %s\n", e.what());
            } catch (...) {
                fprintf(stderr, "[scheduler] run_fn threw unknown\n");
            }
            seq->finished.store(true, std::memory_order_release);
            slots_.release(slot);
        }
    }

    SlotManager   slots_;
    RunFn         run_fn_;
    std::mutex    forward_mu_;
    int           max_queue_ = 0;  // 0 = unbounded

    std::mutex                              queue_mu_;
    std::condition_variable                 queue_cv_;
    std::deque<std::shared_ptr<Sequence>>   queue_;
    bool                                    stop_;
    std::vector<std::thread>                workers_;
};

}  // namespace qwen_engine
