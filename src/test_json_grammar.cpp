// Standalone smoke test for json_grammar.h. Build: g++ -std=c++17 -O0 -o
// /tmp/test_json_grammar src/test_json_grammar.cpp && /tmp/test_json_grammar
//
// Each case feeds a byte stream through JsonGrammar and asserts the final
// (corrupt, done) state. Useful while wiring the grammar into the sampler:
// state-machine bugs surface here, away from the GPU pipeline.

#include "json_grammar.h"
#include <cassert>
#include <cstdio>
#include <string>

using qwen_engine::JsonGrammar;

static int passed = 0, failed = 0;

static void check(const std::string& s, bool expect_done, bool expect_corrupt,
                  const char* label) {
    JsonGrammar g;
    bool any_reject = false;
    for (uint8_t c : s) {
        if (!g.feed(c)) { any_reject = true; break; }
    }
    bool done = g.is_done();
    bool corrupt = g.is_corrupt() || any_reject;
    bool ok = (done == expect_done) && (corrupt == expect_corrupt);
    if (ok) {
        passed++;
        printf("  PASS  %-40s done=%d corrupt=%d\n", label, done, corrupt);
    } else {
        failed++;
        printf("  FAIL  %-40s expected done=%d corrupt=%d, got done=%d corrupt=%d   input=%s\n",
               label, expect_done, expect_corrupt, done, corrupt, s.c_str());
    }
}

static void check_try_feed(const std::string& trial, bool expect_accept,
                           const char* label) {
    JsonGrammar g;
    bool ok = g.try_feed_all(trial.data(), trial.size());
    if (ok == expect_accept) {
        passed++;
        printf("  PASS  %-40s try=%d\n", label, ok);
    } else {
        failed++;
        printf("  FAIL  %-40s expected try=%d, got %d   input=%s\n",
               label, expect_accept, ok, trial.c_str());
    }
}

int main() {
    printf("== Valid documents (expect done=1 corrupt=0) ==\n");
    check("{}",                                true,  false, "empty obj");
    check("[]",                                true,  false, "empty arr");
    check("{\"k\":\"v\"}",                     true,  false, "single kv");
    check("{\"k\":1}",                         true,  false, "kv int");
    check("{\"k\":1.5}",                       true,  false, "kv float");
    check("{\"k\":-3}",                        true,  false, "kv neg int");
    check("{\"k\":1e10}",                      true,  false, "kv exp");
    check("{\"k\":-0.5e-2}",                   true,  false, "kv full num");
    check("{\"k\":true,\"j\":false,\"l\":null}", true, false, "kv literals");
    check("[1,2,3]",                           true,  false, "int arr");
    check("[[1,2],[3,4]]",                     true,  false, "nested arr");
    check("{\"a\":[1,{\"b\":\"c\"}]}",         true,  false, "obj arr obj");
    check("\"hello\"",                         true,  false, "top str");
    // Numbers at top with no terminator stay parked in NUM_INT — not corrupt,
    // not done. Acceptable: JSON-object mode requires `{...}` and any normal
    // payload ends in `}` or `]` which closes upstream frames.
    check("42 ",                               true,  false, "top num + ws");
    check("true",                              true,  false, "top true");
    check("false",                             true,  false, "top false");
    check("null",                              true,  false, "top null");
    check("  {  \"k\"  :  \"v\"  }  ",         true,  false, "ws-padded");
    check("\"esc \\\"q\\\" and \\\\b\"",       true,  false, "str escapes");
    check("\"unicode \\u0041\"",               true,  false, "str unicode");

    printf("\n== Invalid documents (expect done=0 corrupt=1) ==\n");
    check("{",                                 false, false, "incomplete obj");  // not corrupt yet, just incomplete
    check("{\"k\"",                            false, false, "obj key only");
    check("{\"k\":}",                          false, true,  "obj missing val");
    check("[1,]",                              false, true,  "trailing comma arr");
    check("{,}",                               false, true,  "obj starts with comma");
    check("{1:2}",                             false, true,  "obj numeric key");
    check("\"unterminated",                    false, false, "unterminated str (incomplete, no corruption yet)");
    check("xx",                                false, true,  "junk start");
    check("[\"\\x\"]",                         false, true,  "bad escape");
    check("01",                                false, true,  "leading-zero int (0 closes doc, 1 is junk → corrupt)");

    printf("\n== try_feed_all (per-token mask path) ==\n");
    check_try_feed("{",                        true,  "open brace fresh");
    check_try_feed("\"",                       true,  "open quote fresh");
    check_try_feed("hello",                    false, "alpha at top");
    check_try_feed("123",                      true,  "digits at top");

    printf("\n== Snapshot integrity: rejection should restore state ==\n");
    {
        JsonGrammar g;
        for (char c : std::string("{\"k\":")) assert(g.feed(c));
        bool t1 = g.try_feed_all("\"v\"", 3);  // trial valid
        bool t2 = g.try_feed_all("not_a_value", 11);  // trial invalid -> must restore
        // After both trials state should still be ST_OBJ_VALUE waiting for value
        bool ok_tail = g.feed('1') && g.feed('}');
        if (t1 && !t2 && ok_tail && g.is_done() && !g.is_corrupt()) {
            passed++;
            printf("  PASS  snapshot restore on rejected trial\n");
        } else {
            failed++;
            printf("  FAIL  snapshot restore broken (t1=%d t2=%d tail_ok=%d done=%d corrupt=%d)\n",
                   t1, t2, ok_tail, g.is_done(), g.is_corrupt());
        }
    }

    printf("\n== Summary: %d passed, %d failed ==\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
