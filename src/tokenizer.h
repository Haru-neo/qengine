#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <set>
#include "gguf.h"

struct Tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_ranks;  // "tok1 tok2" → rank
    int eos_id = 248046, bos_id = -1;
    int im_start = 248045, im_end = 248046;
    bool is_sentencepiece = false;  // true for Gemma (SPM), false for Qwen (GPT-2 BPE)

    // GPT-2 byte↔unicode mapping
    int byte_to_unicode[256];
    std::unordered_map<int, uint8_t> unicode_to_byte;

    void build_byte_map() {
        std::set<int> printable;
        for (int i = 33; i <= 126; i++) printable.insert(i);
        for (int i = 161; i <= 172; i++) printable.insert(i);
        for (int i = 174; i <= 255; i++) printable.insert(i);

        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (printable.count(b)) {
                byte_to_unicode[b] = b;
            } else {
                byte_to_unicode[b] = 256 + n;
                n++;
            }
            unicode_to_byte[byte_to_unicode[b]] = (uint8_t)b;
        }
    }

    // UTF-8 encode a single codepoint
    static std::string utf8_encode(int cp) {
        std::string s;
        if (cp < 0x80) { s += (char)cp; }
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        return s;
    }

    // Iterate UTF-8 chars, call fn(codepoint) for each
    template<typename F>
    static void for_each_utf8(const std::string& s, F fn) {
        for (size_t i = 0; i < s.size(); ) {
            int cp = 0, len = 1;
            uint8_t c = s[i];
            if (c < 0x80) { cp = c; }
            else if (c < 0xE0) { cp = c & 0x1F; len = 2; }
            else if (c < 0xF0) { cp = c & 0x0F; len = 3; }
            else { cp = c & 0x07; len = 4; }
            for (int j = 1; j < len && i + j < s.size(); j++)
                cp = (cp << 6) | (s[i + j] & 0x3F);
            fn(cp, s.substr(i, len));
            i += len;
        }
    }

    bool load_from_gguf(GGUFFile& gguf) {
        build_byte_map();

        auto it = gguf.meta_str_arr.find("tokenizer.ggml.tokens");
        if (it == gguf.meta_str_arr.end()) { fprintf(stderr, "No tokens in GGUF\n"); return false; }
        vocab = it->second;
        for (int i = 0; i < (int)vocab.size(); i++)
            token_to_id[vocab[i]] = i;

        // Detect tokenizer type
        std::string tok_model = gguf.get_str("tokenizer.ggml.model", "gpt2");
        is_sentencepiece = (tok_model == "llama" || tok_model == "t5" ||
                            tok_model == "gemma" || tok_model == "gemma4");

        // Load merges (BPE only)
        if (!is_sentencepiece) {
            auto mit = gguf.meta_str_arr.find("tokenizer.ggml.merges");
            if (mit != gguf.meta_str_arr.end()) {
                for (int i = 0; i < (int)mit->second.size(); i++)
                    merge_ranks[mit->second[i]] = i;
            }
        }

        eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id", 248046);
        bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id", 2);

        // Find special tokens
        auto find_tok = [&](const std::string& s) -> int {
            auto ti = token_to_id.find(s);
            return ti != token_to_id.end() ? ti->second : -1;
        };
        im_start = find_tok("<|im_start|>");
        im_end = find_tok("<|im_end|>");

        printf("Tokenizer: %zu tokens, %zu merges, eos=%d, bos=%d, type=%s\n",
               vocab.size(), merge_ranks.size(), eos_id, bos_id,
               is_sentencepiece ? "sentencepiece" : "bpe");
        return true;
    }

    // Check if token is a special token (contains < and >)
    bool is_special(int id) const {
        if (id < 0 || id >= (int)vocab.size()) return false;
        auto& v = vocab[id];
        return v.size() >= 3 && v[0] == '<' && v.back() == '>';
    }

    // ============ Decode: SentencePiece (Gemma) ============
    std::string decode_spm(const std::vector<int>& ids) const {
        std::string raw;
        for (int id : ids) {
            if (id < 0 || id >= (int)vocab.size()) continue;
            if (is_special(id)) { raw += vocab[id]; continue; }
            raw += vocab[id];
        }
        // Replace ▁ (U+2581, UTF-8: E2 96 81) with space
        std::string result;
        for (size_t i = 0; i < raw.size(); ) {
            if (i + 2 < raw.size() &&
                (uint8_t)raw[i] == 0xE2 && (uint8_t)raw[i+1] == 0x96 && (uint8_t)raw[i+2] == 0x81) {
                result += ' ';
                i += 3;
            } else {
                result += raw[i++];
            }
        }
        return result;
    }

    // ============ Decode: token IDs → text ============
    std::string decode(const std::vector<int>& ids) const {
        if (is_sentencepiece) return decode_spm(ids);
        std::string result;
        std::string pending;  // accumulate non-special tokens for batch GPT-2 decode

        auto flush = [&]() {
            if (pending.empty()) return;
            for_each_utf8(pending, [&](int cp, const std::string& utf8_char) {
                auto bi = unicode_to_byte.find(cp);
                if (bi != unicode_to_byte.end())
                    result += (char)bi->second;
                else
                    result += utf8_char;  // raw UTF-8 vocab (non-GPT2), pass through
            });
            pending.clear();
        };

        for (int id : ids) {
            if (id < 0 || id >= (int)vocab.size()) continue;
            if (is_special(id)) {
                flush();
                result += vocab[id];  // special tokens pass through as-is
            } else {
                pending += vocab[id];
            }
        }
        flush();
        return result;
    }

    // Decode single token
    std::string decode_token(int id) const {
        if (is_special(id)) return vocab[id];
        return decode({id});
    }

    // UTF-8 streaming helper: extract complete UTF-8 chars, leave partial in buf
    static std::string extract_complete_utf8(std::string& buf) {
        std::string complete;
        size_t last_good = 0;
        size_t i = 0;
        while (i < buf.size()) {
            uint8_t c = buf[i];
            int len;
            if (c < 0x80) len = 1;
            else if (c < 0xC0) { i++; continue; }  // stray continuation byte, skip
            else if (c < 0xE0) len = 2;
            else if (c < 0xF0) len = 3;
            else len = 4;
            if (i + len > buf.size()) break;  // incomplete sequence, keep buffered
            complete += buf.substr(i, len);
            i += len;
            last_good = i;
        }
        buf = buf.substr(last_good);
        return complete;
    }

    // ============ Encode: SentencePiece (Gemma) — greedy longest match ============
    std::vector<int> encode_spm(const std::string& text) const {
        if (text.empty()) return {};
        // Replace spaces with ▁
        std::string spm_text;
        for (char c : text) {
            if (c == ' ') spm_text += "\xe2\x96\x81";  // ▁ U+2581
            else spm_text += c;
        }
        // Greedy longest-match tokenization
        std::vector<int> ids;
        size_t i = 0;
        while (i < spm_text.size()) {
            int best_len = 0, best_id = -1;
            // Try longest match first (up to 32 bytes)
            for (int len = std::min((int)(spm_text.size() - i), 32); len >= 1; len--) {
                auto it = token_to_id.find(spm_text.substr(i, len));
                if (it != token_to_id.end()) { best_len = len; best_id = it->second; break; }
            }
            if (best_id >= 0) {
                ids.push_back(best_id);
                i += best_len;
            } else {
                // Byte fallback: <0xXX>
                char buf[16]; snprintf(buf, sizeof(buf), "<0x%02X>", (uint8_t)spm_text[i]);
                auto bi = token_to_id.find(buf);
                if (bi != token_to_id.end()) ids.push_back(bi->second);
                i++;
            }
        }
        return ids;
    }

    // ============ Encode: text → token IDs ============
    std::vector<int> encode(const std::string& text) const {
        if (is_sentencepiece) return encode_spm(text);
        if (text.empty()) return {};

        // 0. Split text on special tokens first, then BPE-encode the gaps
        //    Special tokens in vocab: <tool_call>, </tool_call>, <think>, </think>,
        //    <tool_response>, </tool_response>, etc.
        std::vector<int> ids;
        size_t pos = 0;
        while (pos < text.size()) {
            // Try to match a special token at current position
            int matched_id = -1;
            size_t matched_len = 0;
            for (auto& [tok_str, tok_id] : token_to_id) {
                if (!is_special(tok_id)) continue;  // only match `<...>`-form specials
                if (tok_str.size() > matched_len && tok_str.size() <= text.size() - pos &&
                    text.compare(pos, tok_str.size(), tok_str) == 0) {
                    matched_id = tok_id;
                    matched_len = tok_str.size();
                }
            }
            if (matched_id >= 0) {
                ids.push_back(matched_id);
                pos += matched_len;
                continue;
            }
            // Find next special token to determine the text chunk boundary
            size_t next_special = text.size();
            for (auto& [tok_str, tok_id] : token_to_id) {
                if (!is_special(tok_id)) continue;
                size_t found = text.find(tok_str, pos);
                if (found != std::string::npos && found < next_special)
                    next_special = found;
            }
            // BPE-encode the text chunk before the next special token
            std::string chunk = text.substr(pos, next_special - pos);
            if (!chunk.empty()) {
                auto raw_chunks = pre_tokenize_raw(chunk);
                for (auto& rc : raw_chunks) {
                    std::string gpt2_str;
                    for (uint8_t b : rc)
                        gpt2_str += utf8_encode(byte_to_unicode[b]);
                    auto chunk_ids = bpe_encode(gpt2_str);
                    ids.insert(ids.end(), chunk_ids.begin(), chunk_ids.end());
                }
            }
            pos = next_special;
        }
        return ids;
    }

    // ============ Chat template: Qwen3.5 format ============
    // force_think: 0=model decides (serve), 1=prefill <think>\n (chat), -1=force skip thinking
    std::vector<int> apply_chat(const std::string& system_msg,
                                 const std::vector<std::pair<std::string, std::string>>& messages,
                                 int force_think = 0) const {
        std::vector<int> ids;

        auto add_text = [&](const std::string& text) {
            auto enc = encode(text);
            ids.insert(ids.end(), enc.begin(), enc.end());
        };

        // System message
        if (!system_msg.empty()) {
            ids.push_back(im_start);
            add_text("system\n" + system_msg);
            ids.push_back(im_end);
            add_text("\n");
        }

        // Messages. Assistant turns from prior rounds have their
        // <think>...</think> reasoning stripped before re-injection —
        // otherwise the model treats earlier chain-of-thought as its
        // *current* working memory and keeps extending it, which causes
        // the "2nd-turn hallucination" where the model re-uses a past
        // reasoning block as if it were this turn's internal state.
        // Qwen's official chat template does this strip; we were missing it.
        auto strip_think = [](const std::string& in) {
            std::string out = in;
            while (true) {
                size_t ts = out.find("<think>");
                if (ts == std::string::npos) break;
                size_t te = out.find("</think>", ts);
                if (te == std::string::npos) {
                    // Unterminated — drop from <think> to end.
                    out = out.substr(0, ts);
                    break;
                }
                out.erase(ts, te + 8 - ts);  // len("</think>") == 8
            }
            // Trim leading whitespace left after strip.
            size_t first = out.find_first_not_of(" \n\r\t");
            return (first == std::string::npos) ? std::string() : out.substr(first);
        };

        for (auto& [role, content] : messages) {
            const std::string& use_content = (role == "assistant")
                ? (strip_think(content)) : content;
            ids.push_back(im_start);
            add_text(role + "\n" + use_content);
            ids.push_back(im_end);
            add_text("\n");
        }

        // Start assistant turn
        ids.push_back(im_start);
        add_text("assistant\n");
        if (force_think == 1) {
            // Force thinking: prefill <think>\n
            auto think_tok = token_to_id.find("<think>");
            if (think_tok != token_to_id.end()) {
                ids.push_back(think_tok->second);
                add_text("\n");
            }
        } else if (force_think == -1) {
            // Force skip thinking: <think>\n\n</think>\n\n (Qwen3 template).
            // Prefer single-token specials; fall back to BPE encoding for
            // tokenizers that don't carry `<think>` as one piece (e.g.
            // distill checkpoints) so the prefill still parses identically.
            auto think_tok = token_to_id.find("<think>");
            auto think_end = token_to_id.find("</think>");
            if (think_tok != token_to_id.end() && think_end != token_to_id.end()) {
                ids.push_back(think_tok->second);
                add_text("\n\n");
                ids.push_back(think_end->second);
                add_text("\n\n");
            } else {
                add_text("<think>\n\n</think>\n\n");
            }
        }
        // force_think == 0: just <|im_start|>assistant\n — model decides
        return ids;
    }

    // Convenience: encode a simple user prompt
    std::vector<int> encode_chat(const std::string& user_msg) const {
        return apply_chat("", {{"user", user_msg}});
    }

    // ============ Chat template: Gemma 4 format ============
    // <bos><start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
    std::vector<int> apply_chat_gemma(const std::string& system_msg,
                                       const std::vector<std::pair<std::string, std::string>>& messages) const {
        // Gemma 4 special tokens by name lookup
        auto find_tok = [&](const std::string& s) -> int {
            auto it = token_to_id.find(s);
            return it != token_to_id.end() ? it->second : -1;
        };
        int start_turn = find_tok("<|turn>");   // 105
        int end_turn = find_tok("<turn|>");     // 106
        int nl_tok = find_tok("\n");            // 107

        std::vector<int> ids;
        ids.push_back(bos_id);  // <bos> = 2

        auto add_turn = [&](const std::string& role, const std::string& content) {
            ids.push_back(start_turn);
            // Encode role + \n + content as text
            auto role_enc = encode(role);
            ids.insert(ids.end(), role_enc.begin(), role_enc.end());
            ids.push_back(nl_tok);
            auto content_enc = encode(content);
            ids.insert(ids.end(), content_enc.begin(), content_enc.end());
            ids.push_back(end_turn);
            ids.push_back(nl_tok);
        };

        // System message as first user turn
        if (!system_msg.empty())
            add_turn("user", system_msg);

        for (auto& [role, content] : messages)
            add_turn(role, content);

        // Start model turn
        ids.push_back(start_turn);
        auto model_enc = encode("model");
        ids.insert(ids.end(), model_enc.begin(), model_enc.end());
        ids.push_back(nl_tok);
        return ids;
    }

private:
    // Unicode character class helpers for raw text (before GPT-2 byte encoding)
    static bool is_unicode_letter(int cp) {
        if (cp >= 'A' && cp <= 'Z') return true;
        if (cp >= 'a' && cp <= 'z') return true;
        if (cp >= 0xC0 && cp <= 0x024F) return true;   // Latin extended
        if (cp >= 0x0370 && cp <= 0x03FF) return true;  // Greek
        if (cp >= 0x0400 && cp <= 0x04FF) return true;  // Cyrillic
        if (cp >= 0x0500 && cp <= 0x052F) return true;
        if (cp >= 0x0600 && cp <= 0x06FF) return true;  // Arabic
        if (cp >= 0x0900 && cp <= 0x097F) return true;  // Devanagari
        if (cp >= 0x3040 && cp <= 0x30FF) return true;  // Japanese
        if (cp >= 0x3400 && cp <= 0x4DBF) return true;  // CJK ext A
        if (cp >= 0x4E00 && cp <= 0x9FFF) return true;  // CJK
        if (cp >= 0xAC00 && cp <= 0xD7AF) return true;  // Korean
        if (cp >= 0xF900 && cp <= 0xFAFF) return true;  // CJK compat
        if (cp >= 0x1100 && cp <= 0x11FF) return true;  // Hangul jamo
        if (cp >= 0x20000 && cp <= 0x2A6DF) return true; // CJK ext B
        return false;
    }
    static bool is_unicode_number(int cp) { return cp >= '0' && cp <= '9'; }
    static bool is_unicode_ws(int cp) { return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'; }
    static bool is_newline_raw(int cp) { return cp == '\n' || cp == '\r'; }

    // Pre-tokenize RAW UTF-8 text using Qwen3.5 regex
    // Applied BEFORE GPT-2 byte encoding
    std::vector<std::string> pre_tokenize_raw(const std::string& text) const {
        // Parse into codepoints
        std::vector<std::pair<int, std::string>> chars;
        for_each_utf8(text, [&](int cp, const std::string& utf8) {
            chars.push_back({cp, utf8});
        });

        std::vector<std::string> chunks;
        size_t i = 0, n = chars.size();

        while (i < n) {
            int cp = chars[i].first;
            bool matched = false;

            // Rule 1: Contractions 's 't 're 've 'm 'll 'd
            if (cp == '\'' && i + 1 < n) {
                int c1 = chars[i+1].first;
                if (c1 == 's' || c1 == 'S' || c1 == 't' || c1 == 'T' ||
                    c1 == 'm' || c1 == 'M' || c1 == 'd' || c1 == 'D') {
                    chunks.push_back(chars[i].second + chars[i+1].second);
                    i += 2; continue;
                }
                if (i + 2 < n) {
                    int c2 = chars[i+2].first;
                    if (((c1|32) == 'r' && (c2|32) == 'e') ||
                        ((c1|32) == 'v' && (c2|32) == 'e') ||
                        ((c1|32) == 'l' && (c2|32) == 'l')) {
                        chunks.push_back(chars[i].second + chars[i+1].second + chars[i+2].second);
                        i += 3; continue;
                    }
                }
            }

            // Rule 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ — optional non-letter/number/newline then letters
            {
                size_t j = i;
                if (j < n && !is_newline_raw(chars[j].first) && !is_unicode_letter(chars[j].first) && !is_unicode_number(chars[j].first) &&
                    j + 1 < n && is_unicode_letter(chars[j+1].first)) {
                    j++;  // consume optional leading char
                }
                size_t start = j;
                while (j < n && is_unicode_letter(chars[j].first)) j++;
                if (j > start || (j > i && j > start)) {
                    std::string chunk;
                    for (size_t k = i; k < j; k++) chunk += chars[k].second;
                    if (!chunk.empty()) { chunks.push_back(chunk); i = j; continue; }
                }
            }

            // Rule 3: \p{N} — single digit
            if (is_unicode_number(cp)) {
                chunks.push_back(chars[i].second);
                i++; continue;
            }

            // Rule 4: ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* — optional space + punct + trailing newlines
            {
                size_t j = i;
                bool has_content = false;
                if (j < n && chars[j].first == ' ' && j + 1 < n &&
                    !is_unicode_ws(chars[j+1].first) && !is_unicode_letter(chars[j+1].first) && !is_unicode_number(chars[j+1].first)) {
                    j++;  // optional space
                }
                size_t ps = j;
                while (j < n && !is_unicode_ws(chars[j].first) && !is_unicode_letter(chars[j].first) && !is_unicode_number(chars[j].first)) {
                    j++;
                    has_content = true;
                }
                if (has_content) {
                    while (j < n && is_newline_raw(chars[j].first)) j++;
                    std::string chunk;
                    for (size_t k = i; k < j; k++) chunk += chars[k].second;
                    chunks.push_back(chunk);
                    i = j; continue;
                }
            }

            // Rule 5: \s*[\r\n]+ — optional whitespace then newlines
            if (is_newline_raw(cp) || (is_unicode_ws(cp))) {
                size_t j = i;
                while (j < n && is_unicode_ws(chars[j].first) && !is_newline_raw(chars[j].first)) j++;
                if (j < n && is_newline_raw(chars[j].first)) {
                    while (j < n && is_newline_raw(chars[j].first)) j++;
                    std::string chunk;
                    for (size_t k = i; k < j; k++) chunk += chars[k].second;
                    chunks.push_back(chunk);
                    i = j; continue;
                }
            }

            // Rule 6: \s+ — whitespace
            if (is_unicode_ws(cp)) {
                std::string chunk;
                while (i < n && is_unicode_ws(chars[i].first)) {
                    chunk += chars[i].second;
                    i++;
                }
                chunks.push_back(chunk);
                continue;
            }

            // Fallback
            chunks.push_back(chars[i].second);
            i++;
        }
        return chunks;
    }

    // BPE encode a single pre-tokenized chunk
    std::vector<int> bpe_encode(const std::string& word) const {
        // Split into UTF-8 characters (each is a BPE symbol)
        std::vector<std::string> symbols;
        for_each_utf8(word, [&](int, const std::string& utf8) {
            symbols.push_back(utf8);
        });

        if (symbols.empty()) return {};

        // Repeatedly find and apply highest-priority (lowest rank) merge
        while (symbols.size() > 1) {
            int best_rank = INT_MAX, best_pos = -1;
            for (int i = 0; i + 1 < (int)symbols.size(); i++) {
                std::string pair = symbols[i] + " " + symbols[i + 1];
                auto it = merge_ranks.find(pair);
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                }
            }
            if (best_pos < 0) break;
            symbols[best_pos] += symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);
        }

        // Convert to IDs
        std::vector<int> ids;
        for (auto& s : symbols) {
            auto ti = token_to_id.find(s);
            if (ti != token_to_id.end()) {
                ids.push_back(ti->second);
            } else {
                // Byte fallback
                for (uint8_t c : s) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "<0x%02X>", c);
                    auto bi = token_to_id.find(buf);
                    if (bi != token_to_id.end()) ids.push_back(bi->second);
                }
            }
        }
        return ids;
    }
};
