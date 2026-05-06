// Constrained-decoding grammar for OpenAI `response_format`.
//
// Supports:
//   - {"type": "json_object"}  → output must be a syntactically valid JSON value
//   - {"type": "json_schema", ...} → currently treated as json_object;
//     schema-aware validation is a follow-up.
//
// Design: a stack-based JSON state machine. `feed(byte)` advances state; on a
// disallowed byte the grammar marks itself corrupt and `feed` returns false.
// `try_feed_all(bytes)` snapshots state, attempts the run, and restores on
// failure — used per candidate token to decide if that token is permissible
// in the current state. After sampling, `feed` is called for each byte of the
// chosen token to commit the transition.
//
// Special tokens (those whose decoded string starts with `<` and ends with
// `>`, e.g. `<|im_end|>`) are exempted by the caller and always pass through.
//
// Cost: per-step we walk every (id) in the vocab and run `try_feed_all`. With
// ~250K vocab × ~5 bytes/token × ~30 ns/byte ≈ 40 ms on the structural steps.
// Most decoding steps are mid-string, where the inner check is a single byte
// class test, so the actual cost is closer to 5 ms.

#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace qwen_engine {

struct JsonGrammar {
    enum State : uint8_t {
        ST_TOP_VALUE,        // before any top-level value
        ST_TOP_DONE,         // after top-level value (only ws allowed)
        ST_OBJ_KEY_OR_END,   // after `{`: expect `"` or `}`
        ST_OBJ_KEY_START,    // after `,` in obj: expect `"`
        ST_OBJ_AFTER_KEY,    // after closing `"` of key: expect `:`
        ST_OBJ_VALUE,        // after `:`: expect value
        ST_OBJ_AFTER_VAL,    // after value in obj: expect `,` or `}`
        ST_ARR_VAL_OR_END,   // after `[`: expect value or `]`
        ST_ARR_AFTER_VAL,    // after value in arr: expect `,` or `]`
        ST_ARR_AFTER_COMMA,  // after `,` in arr: expect value
        ST_STR_BODY,
        ST_STR_ESC,
        ST_STR_U0, ST_STR_U1, ST_STR_U2, ST_STR_U3,
        ST_NUM_AFTER_MINUS,
        ST_NUM_AFTER_ZERO,   // saw "0" or "-0", needs `.` or `e/E` or terminator
        ST_NUM_INT,
        ST_NUM_AFTER_DOT,
        ST_NUM_FRAC,
        ST_NUM_AFTER_E,
        ST_NUM_EXP_SIGN,
        ST_NUM_EXP,
        ST_LIT_T1, ST_LIT_T2, ST_LIT_T3,
        ST_LIT_F1, ST_LIT_F2, ST_LIT_F3, ST_LIT_F4,
        ST_LIT_N1, ST_LIT_N2, ST_LIT_N3,
    };

    // Stack of states. Top is current. The bottom is always ST_TOP_VALUE
    // (later replaced by ST_TOP_DONE).
    std::vector<State> stack;
    // Parallel stack for STR frames: true if this string is a key, false if a
    // value. Lets us route correctly on `"` close.
    std::vector<uint8_t> str_is_key;
    bool corrupt = false;

    JsonGrammar() { reset(); }

    void reset() {
        stack.clear();
        str_is_key.clear();
        stack.push_back(ST_TOP_VALUE);
        corrupt = false;
    }

    bool is_corrupt() const { return corrupt; }
    bool is_done() const { return !corrupt && stack.size() == 1 && stack[0] == ST_TOP_DONE; }

    static bool is_ws(uint8_t b)  { return b == ' ' || b == '\t' || b == '\n' || b == '\r'; }
    static bool is_dig(uint8_t b) { return b >= '0' && b <= '9'; }
    static bool is_hex(uint8_t b) {
        return is_dig(b) || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F');
    }

    State top() const { return stack.back(); }
    void  set_top(State s) { stack.back() = s; }
    void  push(State s) { stack.push_back(s); }

    // Pure read-only trial: simulate feeding all bytes, restore state in both
    // outcomes. Used by the sampler to mask a candidate token without
    // advancing the grammar — the chosen token is later committed via
    // explicit feed() calls.
    bool try_feed_all(const char* p, size_t n) {
        // Snapshot is cheap: stack rarely exceeds depth 8 in practice.
        auto saved_stack = stack;
        auto saved_keys  = str_is_key;
        bool saved_corrupt = corrupt;
        bool ok = true;
        for (size_t i = 0; i < n; i++) {
            if (!feed((uint8_t)p[i])) { ok = false; break; }
        }
        // Always restore — try_feed_all has no side effects on state.
        stack = std::move(saved_stack);
        str_is_key = std::move(saved_keys);
        corrupt = saved_corrupt;
        return ok;
    }

    // Permanently consume a single byte. Returns false on disallowed input
    // (and marks the grammar corrupt; caller usually restores via try_feed_all).
    bool feed(uint8_t b) {
        if (corrupt) return false;
        State s = top();

        // Whitespace skip in structural states only.
        const bool is_struct =
            s == ST_TOP_VALUE || s == ST_TOP_DONE ||
            s == ST_OBJ_KEY_OR_END || s == ST_OBJ_KEY_START ||
            s == ST_OBJ_AFTER_KEY  || s == ST_OBJ_VALUE ||
            s == ST_OBJ_AFTER_VAL  ||
            s == ST_ARR_VAL_OR_END || s == ST_ARR_AFTER_VAL || s == ST_ARR_AFTER_COMMA;
        if (is_struct && is_ws(b)) return true;

        switch (s) {
            case ST_TOP_VALUE:    return start_value(b);
            case ST_TOP_DONE:     corrupt = true; return false;

            case ST_OBJ_KEY_OR_END:
                if (b == '}') { stack.pop_back(); return promote_after_value(); }
                if (b == '"') { set_top(ST_OBJ_AFTER_KEY); push(ST_STR_BODY); str_is_key.push_back(1); return true; }
                corrupt = true; return false;

            case ST_OBJ_KEY_START:
                if (b == '"') { set_top(ST_OBJ_AFTER_KEY); push(ST_STR_BODY); str_is_key.push_back(1); return true; }
                corrupt = true; return false;

            case ST_OBJ_AFTER_KEY:
                if (b == ':') { set_top(ST_OBJ_VALUE); return true; }
                corrupt = true; return false;

            case ST_OBJ_VALUE:    return start_value(b);

            case ST_OBJ_AFTER_VAL:
                if (b == ',') { set_top(ST_OBJ_KEY_START); return true; }
                if (b == '}') { stack.pop_back(); return promote_after_value(); }
                corrupt = true; return false;

            case ST_ARR_VAL_OR_END:
                if (b == ']') { stack.pop_back(); return promote_after_value(); }
                return start_value(b);
            case ST_ARR_AFTER_COMMA:
                return start_value(b);
            case ST_ARR_AFTER_VAL:
                if (b == ',') { set_top(ST_ARR_AFTER_COMMA); return true; }
                if (b == ']') { stack.pop_back(); return promote_after_value(); }
                corrupt = true; return false;

            case ST_STR_BODY:
                if (b == '"') {
                    stack.pop_back();
                    bool was_key = str_is_key.back(); str_is_key.pop_back();
                    if (was_key) return true;  // parent is already ST_OBJ_AFTER_KEY
                    return promote_after_value();
                }
                if (b == '\\') { set_top(ST_STR_ESC); return true; }
                if (b < 0x20) { corrupt = true; return false; }
                return true;
            case ST_STR_ESC:
                if (b == '"' || b == '\\' || b == '/' || b == 'b' || b == 'f' ||
                    b == 'n' || b == 'r' || b == 't') { set_top(ST_STR_BODY); return true; }
                if (b == 'u') { set_top(ST_STR_U0); return true; }
                corrupt = true; return false;
            case ST_STR_U0: if (is_hex(b)) { set_top(ST_STR_U1); return true; } corrupt = true; return false;
            case ST_STR_U1: if (is_hex(b)) { set_top(ST_STR_U2); return true; } corrupt = true; return false;
            case ST_STR_U2: if (is_hex(b)) { set_top(ST_STR_U3); return true; } corrupt = true; return false;
            case ST_STR_U3: if (is_hex(b)) { set_top(ST_STR_BODY); return true; } corrupt = true; return false;

            case ST_NUM_AFTER_MINUS:
                if (b == '0') { set_top(ST_NUM_AFTER_ZERO); return true; }
                if (b >= '1' && b <= '9') { set_top(ST_NUM_INT); return true; }
                corrupt = true; return false;
            case ST_NUM_AFTER_ZERO:
                if (b == '.') { set_top(ST_NUM_AFTER_DOT); return true; }
                if (b == 'e' || b == 'E') { set_top(ST_NUM_AFTER_E); return true; }
                stack.pop_back();
                if (!promote_after_value()) return false;
                return feed(b);
            case ST_NUM_INT:
                if (is_dig(b)) return true;
                if (b == '.') { set_top(ST_NUM_AFTER_DOT); return true; }
                if (b == 'e' || b == 'E') { set_top(ST_NUM_AFTER_E); return true; }
                stack.pop_back();
                if (!promote_after_value()) return false;
                return feed(b);
            case ST_NUM_AFTER_DOT:
                if (is_dig(b)) { set_top(ST_NUM_FRAC); return true; }
                corrupt = true; return false;
            case ST_NUM_FRAC:
                if (is_dig(b)) return true;
                if (b == 'e' || b == 'E') { set_top(ST_NUM_AFTER_E); return true; }
                stack.pop_back();
                if (!promote_after_value()) return false;
                return feed(b);
            case ST_NUM_AFTER_E:
                if (b == '+' || b == '-') { set_top(ST_NUM_EXP_SIGN); return true; }
                if (is_dig(b)) { set_top(ST_NUM_EXP); return true; }
                corrupt = true; return false;
            case ST_NUM_EXP_SIGN:
                if (is_dig(b)) { set_top(ST_NUM_EXP); return true; }
                corrupt = true; return false;
            case ST_NUM_EXP:
                if (is_dig(b)) return true;
                stack.pop_back();
                if (!promote_after_value()) return false;
                return feed(b);

            case ST_LIT_T1: if (b == 'r') { set_top(ST_LIT_T2); return true; } break;
            case ST_LIT_T2: if (b == 'u') { set_top(ST_LIT_T3); return true; } break;
            case ST_LIT_T3: if (b == 'e') { stack.pop_back(); return promote_after_value(); } break;
            case ST_LIT_F1: if (b == 'a') { set_top(ST_LIT_F2); return true; } break;
            case ST_LIT_F2: if (b == 'l') { set_top(ST_LIT_F3); return true; } break;
            case ST_LIT_F3: if (b == 's') { set_top(ST_LIT_F4); return true; } break;
            case ST_LIT_F4: if (b == 'e') { stack.pop_back(); return promote_after_value(); } break;
            case ST_LIT_N1: if (b == 'u') { set_top(ST_LIT_N2); return true; } break;
            case ST_LIT_N2: if (b == 'l') { set_top(ST_LIT_N3); return true; } break;
            case ST_LIT_N3: if (b == 'l') { stack.pop_back(); return promote_after_value(); } break;
        }
        corrupt = true;
        return false;
    }

    // Begin a value at the current state. Pushes a new frame.
    bool start_value(uint8_t b) {
        if (b == '{') { push(ST_OBJ_KEY_OR_END); return true; }
        if (b == '[') { push(ST_ARR_VAL_OR_END); return true; }
        if (b == '"') { push(ST_STR_BODY); str_is_key.push_back(0); return true; }
        if (b == '-') { push(ST_NUM_AFTER_MINUS); return true; }
        if (b == '0') { push(ST_NUM_AFTER_ZERO); return true; }
        if (b >= '1' && b <= '9') { push(ST_NUM_INT); return true; }
        if (b == 't') { push(ST_LIT_T1); return true; }
        if (b == 'f') { push(ST_LIT_F1); return true; }
        if (b == 'n') { push(ST_LIT_N1); return true; }
        corrupt = true;
        return false;
    }

    // After a value (object/array/string/number/literal) closes, advance the
    // newly exposed top to the appropriate "after value" state.
    bool promote_after_value() {
        if (stack.empty()) { corrupt = true; return false; }
        State& t = stack.back();
        switch (t) {
            case ST_TOP_VALUE:                       t = ST_TOP_DONE;       return true;
            case ST_OBJ_VALUE:                       t = ST_OBJ_AFTER_VAL;  return true;
            case ST_ARR_VAL_OR_END:
            case ST_ARR_AFTER_COMMA:                 t = ST_ARR_AFTER_VAL;  return true;
            case ST_TOP_DONE:                        return true;  // nested pop after `]`/`}` already in DONE
            case ST_OBJ_AFTER_VAL:
            case ST_ARR_AFTER_VAL:                   return true;  // nested container close
            default:                                 corrupt = true; return false;
        }
    }
};

}  // namespace qwen_engine
