// Unit test for the <think> / reasoning_content split (server.h).
//
// Covers the case that produced issue #4: enable_thinking prefills an opening
// tag into the prompt, so the generated stream starts *inside* the block and
// the reply ends with a bare </think>.
//
//   g++ -std=c++17 -I../src -o /tmp/test_think_split test_think_split.cpp && /tmp/test_think_split
#include "sampling.h"
#include "server.h"
#include <cstdio>

static int failures = 0;

static void eq(const char* what, const std::string& got, const std::string& want) {
    if (got == want) return;
    printf("  FAIL %-28s got=[%s] want=[%s]\n", what, got.c_str(), want.c_str());
    failures++;
}
static void eqb(const char* what, bool got, bool want) {
    if (got == want) return;
    printf("  FAIL %-28s got=%d want=%d\n", what, (int)got, (int)want);
    failures++;
}

// Drive the streaming splitter one byte at a time — the worst case for tags
// straddling chunk boundaries.
static void stream_bytes(const std::string& text, bool prefilled,
                         std::string& reasoning, std::string& content) {
    ThinkStreamSplitter s(prefilled);
    for (char c : text) s.feed(std::string(1, c), false, reasoning, content);
    s.feed("", true, reasoning, content);
}

// Same text, but chunked in 5s (tokens are multi-byte in practice).
static void stream_chunks(const std::string& text, bool prefilled, size_t n,
                          std::string& reasoning, std::string& content) {
    ThinkStreamSplitter s(prefilled);
    for (size_t i = 0; i < text.size(); i += n)
        s.feed(text.substr(i, n), false, reasoning, content);
    s.feed("", true, reasoning, content);
}

static void check(const char* name, const std::string& text, bool prefilled,
                  const std::string& want_reasoning, const std::string& want_content,
                  bool want_closed) {
    printf("%s\n", name);
    ThinkSplit b = split_think(text, prefilled);
    eq("batch.reasoning", b.reasoning, want_reasoning);
    eq("batch.content", b.content, want_content);
    eqb("batch.closed", b.closed, want_closed);

    // Streaming must reassemble to exactly the same bytes, at any chunking.
    std::string r1, c1, r2, c2;
    stream_bytes(text, prefilled, r1, c1);
    stream_chunks(text, prefilled, 5, r2, c2);
    eq("stream1B.reasoning", r1, want_reasoning);
    eq("stream1B.content", c1, want_content);
    eq("stream5B.reasoning", r2, want_reasoning);
    eq("stream5B.content", c2, want_content);
}

int main() {
    // issue #4: enable_thinking=true. Prompt ends with "<think>\n", so the
    // model resumes mid-reasoning and closes with a bare tag.
    check("prefilled, closed", "Let me think.\n</think>\n\nHello!", true,
          "Let me think.", "Hello!", true);

    // Cut off at max_tokens before the model ever closed the block. The old
    // strip_think_block dropped the whole reply here.
    check("prefilled, never closed", "Still reasoning and then the budget ran", true,
          "Still reasoning and then the budget ran", "", false);

    // force_think == 0: the model opens its own tag.
    check("model-opened block", "<think>\nreasoning here\n</think>\n\nThe answer.", false,
          "reasoning here", "The answer.", true);

    // force_think == -1 (json mode): the closed pair is in the *prompt*, so
    // generation is pure content and nothing must be swallowed.
    check("no think at all", "{\"ok\": true}", false, "", "{\"ok\": true}", true);

    // Content that merely mentions the tag characters must not be eaten.
    check("angle brackets in answer", "a < b and c > d", false, "", "a < b and c > d", true);

    // A second block mid-answer still routes to reasoning, not content.
    check("two blocks", "<think>one</think>alpha<think>two</think>beta", false,
          "one\ntwo", "alphabeta", true);

    // Unterminated model-opened block: keep what we have. Content keeps its
    // trailing space — only leading whitespace is trimmed there.
    check("model-opened, unclosed", "prelude <think>tail with no close", false,
          "tail with no close", "prelude ", false);

    // Nothing but reasoning, closed, empty answer (model stopped right after).
    check("empty answer", "done\n</think>", true, "done", "", true);

    printf(failures ? "\n%d FAILED\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
