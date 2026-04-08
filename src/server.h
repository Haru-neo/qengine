#pragma once
#include <string>
#include <vector>
#include <functional>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// ============ Minimal JSON helpers ============

static std::string json_escape(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ) {
        uint8_t c = s[i];
        if (c >= 0x80) {
            // Multi-byte UTF-8: pass through intact
            int len = (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            for (int j = 0; j < len && i < s.size(); j++, i++)
                out += s[i];
        } else {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += (char)c;
                    }
            }
            i++;
        }
    }
    return out;
}

// Extract raw JSON value (object/array) for a key
static std::string json_get_raw(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    if (pos >= json.size()) return "";
    char open = json[pos];
    if (open == '[' || open == '{') {
        char close = (open == '[') ? ']' : '}';
        int depth = 0;
        bool in_str = false;
        size_t start = pos;
        for (size_t i = pos; i < json.size(); i++) {
            if (in_str) {
                if (json[i] == '\\') { i++; continue; }
                if (json[i] == '"') in_str = false;
            } else {
                if (json[i] == '"') in_str = true;
                else if (json[i] == open) depth++;
                else if (json[i] == close) { depth--; if (depth == 0) return json.substr(start, i - start + 1); }
            }
        }
    }
    return "";
}

// Extract string value for a key from JSON (simple, not recursive)
static std::string json_get_str(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    pos++;
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case '/': result += '/'; break;
                case 'u': {
                    // \uXXXX → UTF-8
                    if (pos + 4 < json.size()) {
                        uint32_t cp = strtoul(json.substr(pos+1, 4).c_str(), nullptr, 16);
                        pos += 4;
                        // Handle surrogate pairs \uD800-\uDBFF \uDC00-\uDFFF
                        if (cp >= 0xD800 && cp <= 0xDBFF && pos + 2 < json.size()
                            && json[pos+1] == '\\' && json[pos+2] == 'u') {
                            uint32_t lo = strtoul(json.substr(pos+3, 4).c_str(), nullptr, 16);
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                pos += 6;
                            }
                        }
                        // Encode as UTF-8
                        if (cp < 0x80) { result += (char)cp; }
                        else if (cp < 0x800) { result += (char)(0xC0|(cp>>6)); result += (char)(0x80|(cp&0x3F)); }
                        else if (cp < 0x10000) { result += (char)(0xE0|(cp>>12)); result += (char)(0x80|((cp>>6)&0x3F)); result += (char)(0x80|(cp&0x3F)); }
                        else { result += (char)(0xF0|(cp>>18)); result += (char)(0x80|((cp>>12)&0x3F)); result += (char)(0x80|((cp>>6)&0x3F)); result += (char)(0x80|(cp&0x3F)); }
                    }
                    break;
                }
                default: result += json[pos];
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

static int json_get_int(const std::string& json, const std::string& key, int def = 0) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    return atoi(json.c_str() + pos);
}

static float json_get_float(const std::string& json, const std::string& key, float def = 0.0f) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    return (float)atof(json.c_str() + pos);
}

// Normalize JSON whitespace: ensure space after : and , (matches Jinja tojson output)
static std::string json_normalize(const std::string& json) {
    std::string r;
    bool in_str = false;
    for (size_t i = 0; i < json.size(); i++) {
        char c = json[i];
        r += c;
        if (in_str) {
            if (c == '\\' && i + 1 < json.size()) { r += json[++i]; continue; }
            if (c == '"') in_str = false;
        } else {
            if (c == '"') in_str = true;
            else if ((c == ':' || c == ',') && i + 1 < json.size() && json[i + 1] != ' ')
                r += ' ';
        }
    }
    return r;
}

// Find matching '}' respecting nesting and string escapes
static size_t find_obj_end(const std::string& json, size_t start) {
    int depth = 0;
    bool in_str = false;
    for (size_t i = start; i < json.size(); i++) {
        char c = json[i];
        if (in_str) {
            if (c == '\\') { i++; continue; }  // skip escaped char
            if (c == '"') in_str = false;
        } else {
            if (c == '"') in_str = true;
            else if (c == '{') depth++;
            else if (c == '}') { depth--; if (depth == 0) return i; }
        }
    }
    return std::string::npos;
}

// ============ Tool call helpers ============

// Iterate objects in JSON array, returns each {...} substring
static std::vector<std::string> json_array_objects(const std::string& arr) {
    std::vector<std::string> objs;
    size_t pos = arr.find('[');
    if (pos == std::string::npos) return objs;
    while (true) {
        size_t start = arr.find('{', pos);
        if (start == std::string::npos) break;
        size_t end = find_obj_end(arr, start);
        if (end == std::string::npos) break;
        objs.push_back(arr.substr(start, end - start + 1));
        pos = end + 1;
        size_t next_brace = arr.find('{', pos);
        size_t arr_end = arr.find(']', pos);
        if (arr_end < next_brace || next_brace == std::string::npos) break;
    }
    return objs;
}

// Parse top-level key→value pairs from JSON object (handles string/number/bool/array/object)
struct JsonKV { std::string key; std::string value; };
static std::vector<JsonKV> json_flat_kv(const std::string& json) {
    std::vector<JsonKV> result;
    int depth = 0; bool in_str = false;
    for (size_t i = 0; i < json.size(); i++) {
        char c = json[i];
        if (in_str) { if (c == '\\') { i++; continue; } if (c == '"') in_str = false; continue; }
        if (c == '"' && depth == 1) {
            size_t kend = json.find('"', i + 1);
            if (kend == std::string::npos) break;
            std::string key = json.substr(i + 1, kend - i - 1);
            size_t colon = json.find(':', kend + 1);
            if (colon == std::string::npos) break;
            size_t vs = colon + 1;
            while (vs < json.size() && (json[vs] == ' ' || json[vs] == '\n')) vs++;
            if (vs >= json.size()) break;
            std::string val;
            if (json[vs] == '"') {
                val = json_get_str(json, key);
                size_t ve = vs + 1;
                while (ve < json.size()) { if (json[ve] == '\\') { ve += 2; continue; } if (json[ve] == '"') break; ve++; }
                i = ve;
            } else if (json[vs] == '{') {
                size_t ve = find_obj_end(json, vs);
                if (ve != std::string::npos) { val = json.substr(vs, ve - vs + 1); i = ve; }
            } else if (json[vs] == '[') {
                int d = 0; bool is2 = false; size_t ve = vs;
                for (; ve < json.size(); ve++) {
                    if (is2) { if (json[ve] == '\\') { ve++; continue; } if (json[ve] == '"') is2 = false; continue; }
                    if (json[ve] == '"') is2 = true;
                    else if (json[ve] == '[') d++;
                    else if (json[ve] == ']') { d--; if (d == 0) break; }
                }
                val = json.substr(vs, ve - vs + 1); i = ve;
            } else {
                size_t ve = json.find_first_of(",}] \n\r\t", vs);
                if (ve == std::string::npos) ve = json.size();
                val = json.substr(vs, ve - vs); i = ve - 1;
            }
            result.push_back({key, val});
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{' || c == '[') depth++;
        else if (c == '}' || c == ']') depth--;
    }
    return result;
}

// Convenience: same as json_flat_kv but only returns entries where value is an object
static std::vector<JsonKV> json_obj_entries(const std::string& json) {
    auto all = json_flat_kv(json);
    std::vector<JsonKV> result;
    for (auto& kv : all)
        if (!kv.value.empty() && kv.value[0] == '{') result.push_back(kv);
    return result;
}

// Convert OpenAI tools JSON → Qwen3.5 XML format (matching Qwen3-Coder.jinja exactly)
static std::string tools_json_to_xml(const std::string& tools_json) {
    std::string xml;
    auto tools = json_array_objects(tools_json);
    for (auto& tool_obj : tools) {
        std::string func_json = json_get_raw(tool_obj, "function");
        if (func_json.empty()) func_json = tool_obj;

        std::string name = json_get_str(func_json, "name");
        std::string desc = json_get_str(func_json, "description");
        std::string params_json = json_get_raw(func_json, "parameters");

        xml += "\n<function>\n<name>" + name + "</name>";
        if (!desc.empty()) xml += "\n<description>" + desc + "</description>";
        xml += "\n<parameters>";

        if (!params_json.empty()) {
            std::string props_json = json_get_raw(params_json, "properties");
            if (!props_json.empty()) {
                auto entries = json_obj_entries(props_json);  // only object-valued entries
                for (auto& e : entries) {
                    xml += "\n<parameter>\n<name>" + e.key + "</name>";
                    std::string ptype = json_get_str(e.value, "type");
                    std::string pdesc = json_get_str(e.value, "description");
                    std::string penum = json_get_raw(e.value, "enum");
                    if (!ptype.empty()) xml += "\n<type>" + ptype + "</type>";
                    if (!pdesc.empty()) xml += "\n<description>" + pdesc + "</description>";
                    if (!penum.empty()) xml += "\n<enum>" + penum + "</enum>";
                    xml += "\n</parameter>";
                }
            }
            std::string req = json_get_raw(params_json, "required");
            if (!req.empty()) xml += "\n<required>" + req + "</required>";
        }
        xml += "\n</parameters>\n</function>";
    }
    return xml;
}

// Convert assistant tool_calls JSON → Qwen3.5 XML for prompt
// Input: raw JSON string of tool_calls array from OpenAI API
static std::string tool_calls_to_xml(const std::string& tc_json) {
    std::string xml;
    auto calls = json_array_objects(tc_json);
    for (auto& call_obj : calls) {
        std::string func_json = json_get_raw(call_obj, "function");
        if (func_json.empty()) continue;
        std::string fname = json_get_str(func_json, "name");
        std::string args_str = json_get_str(func_json, "arguments");

        xml += "\n<tool_call>\n<function=" + fname + ">\n";
        if (!args_str.empty()) {
            auto kvs = json_flat_kv(args_str);  // parse {"key":"val", ...}
            for (auto& kv : kvs) {
                xml += "<parameter=" + kv.key + ">\n" + kv.value + "\n</parameter>\n";
            }
        }
        xml += "</function>\n</tool_call>";
    }
    return xml;
}

// Parse <tool_call> XML from model output (gist template format)
// Format: <tool_call>\n<function=name>\n<parameter=key>\nval\n</parameter>\n</function>\n</tool_call>
// Also handles JSON format as fallback: <tool_call>\n{"name":...}\n</tool_call>
struct ParsedToolCall {
    std::string name;
    std::string arguments;  // JSON arguments string
};

static std::vector<ParsedToolCall> parse_tool_calls(const std::string& text) {
    std::vector<ParsedToolCall> calls;
    size_t pos = 0;
    while (true) {
        size_t tc_start = text.find("<tool_call>", pos);
        if (tc_start == std::string::npos) break;
        size_t tc_end = text.find("</tool_call>", tc_start);
        if (tc_end == std::string::npos) tc_end = text.size();
        std::string block = text.substr(tc_start + 11, tc_end - tc_start - 11);

        ParsedToolCall call;
        // Try XML format first: <function=NAME>...<parameter=KEY>VAL</parameter>...</function>
        size_t fn_start = block.find("<function=");
        if (fn_start != std::string::npos) {
            fn_start += 10;
            size_t fn_end = block.find('>', fn_start);
            if (fn_end != std::string::npos) call.name = block.substr(fn_start, fn_end - fn_start);
            // Extract parameters → build JSON args
            std::string args = "{";
            size_t pp = 0; int pcount = 0;
            while (true) {
                size_t ps = block.find("<parameter=", pp);
                if (ps == std::string::npos) break;
                ps += 11;
                size_t pe = block.find('>', ps);
                if (pe == std::string::npos) break;
                std::string key = block.substr(ps, pe - ps);
                size_t vs = pe + 1;
                if (vs < block.size() && block[vs] == '\n') vs++;
                size_t ve = block.find("</parameter>", vs);
                if (ve == std::string::npos) break;
                std::string val = block.substr(vs, ve - vs);
                while (!val.empty() && val.back() == '\n') val.pop_back();
                if (pcount > 0) args += ",";
                args += "\"" + json_escape(key) + "\":\"" + json_escape(val) + "\"";
                pcount++;
                pp = ve + 12;
            }
            args += "}";
            call.arguments = args;
        } else {
            // Fallback: JSON format {"name":..., "arguments":...}
            size_t js = block.find('{');
            if (js != std::string::npos) {
                size_t je = block.rfind('}');
                if (je != std::string::npos && je > js) {
                    std::string json_str = block.substr(js, je - js + 1);
                    call.name = json_get_str(json_str, "name");
                    call.arguments = json_get_raw(json_str, "arguments");
                }
            }
        }
        if (!call.name.empty()) calls.push_back(call);
        pos = tc_end + 12;
    }
    return calls;
}

// Build OpenAI-format tool_calls JSON array
static std::string tool_calls_to_api_json(const std::vector<ParsedToolCall>& calls) {
    std::string json = "[";
    for (size_t i = 0; i < calls.size(); i++) {
        auto& c = calls[i];
        if (i > 0) json += ",";
        json += "{\"id\":\"call_" + std::to_string(i) + "\",\"type\":\"function\","
              + "\"function\":{\"name\":\"" + json_escape(c.name) + "\","
              + "\"arguments\":\"" + json_escape(c.arguments) + "\"}}";
    }
    json += "]";
    return json;
}

// Strip <think>...</think> from content, return cleaned content
static std::string strip_think_block(const std::string& text) {
    std::string result = text;
    while (true) {
        size_t ts = result.find("<think>");
        if (ts == std::string::npos) break;
        size_t te = result.find("</think>", ts);
        if (te == std::string::npos) {
            result = result.substr(0, ts);  // unclosed think = strip rest
            break;
        }
        result = result.substr(0, ts) + result.substr(te + 8);
    }
    // Trim leading whitespace
    size_t first = result.find_first_not_of(" \n\r\t");
    if (first != std::string::npos) result = result.substr(first);
    else result.clear();
    return result;
}

// Strip tool_call XML blocks from content
static std::string strip_tool_calls(const std::string& text) {
    std::string result = text;
    while (true) {
        size_t ts = result.find("<tool_call>");
        if (ts == std::string::npos) break;
        size_t te = result.find("</tool_call>", ts);
        if (te == std::string::npos) {
            result = result.substr(0, ts);
            break;
        }
        result = result.substr(0, ts) + result.substr(te + 12);
    }
    // Trim trailing whitespace
    while (!result.empty() && (result.back() == ' ' || result.back() == '\n'))
        result.pop_back();
    return result;
}

// Extract messages array from chat completions request
struct ChatMessage {
    std::string role;
    std::string content;
    std::string tool_calls_json;  // raw JSON array for assistant tool_calls
};

static std::vector<ChatMessage> json_get_messages(const std::string& json) {
    std::vector<ChatMessage> msgs;
    size_t pos = json.find("\"messages\"");
    if (pos == std::string::npos) return msgs;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return msgs;

    while (true) {
        size_t obj_start = json.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end = find_obj_end(json, obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj = json.substr(obj_start, obj_end - obj_start + 1);
        ChatMessage msg;
        msg.role = json_get_str(obj, "role");
        msg.content = json_get_str(obj, "content");
        // Parse tool_calls array from assistant messages
        msg.tool_calls_json = json_get_raw(obj, "tool_calls");
        if (!msg.role.empty()) msgs.push_back(msg);

        pos = obj_end + 1;
        if (json.find(']', pos) < json.find('{', pos)) break;
    }
    return msgs;
}

// ============ HTTP Server ============

using GenerateFunc = std::function<std::string(const std::vector<int>& prompt_ids, int max_tokens)>;
using StreamCallback = std::function<void(const std::string& token_text, bool is_done)>;
using StreamGenerateFunc = std::function<void(const std::vector<int>& prompt_ids, int max_tokens, StreamCallback cb)>;

struct HttpServer {
    int port;
    int server_fd = -1;
    GenerateFunc generate_fn;
    SamplingParams* sampling_params = nullptr;  // pointer to shared sampling params
    StreamGenerateFunc stream_generate_fn;
    std::function<std::vector<int>(const std::string&)> encode_fn;
    std::function<std::vector<int>(const std::vector<std::pair<std::string,std::string>>&)> chat_encode_fn;
    std::string model_name = "qwen";
    std::string api_key;

    void send_response(int client_fd, int status, const std::string& content_type, const std::string& body) {
        std::string status_text = (status == 200) ? "OK" : "Bad Request";
        std::ostringstream resp;
        resp << "HTTP/1.1 " << status << " " << status_text << "\r\n"
             << "Content-Type: " << content_type << "\r\n"
             << "Content-Length: " << body.size() << "\r\n"
             << "Access-Control-Allow-Origin: *\r\n"
             << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
             << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
             << "Connection: close\r\n"
             << "\r\n"
             << body;
        std::string r = resp.str();
        ::send(client_fd, r.c_str(), r.size(), 0);
    }

    void handle_models(int client_fd) {
        std::ostringstream json;
        json << "{\"object\":\"list\",\"data\":[{\"id\":\"" << model_name
             << "\",\"object\":\"model\",\"owned_by\":\"local\"}]}";
        send_response(client_fd, 200, "application/json", json.str());
    }

    void send_sse(int fd, const std::string& data) {
        std::string chunk = "data: " + data + "\n\n";
        ::send(fd, chunk.c_str(), chunk.size(), MSG_NOSIGNAL);
    }

    void handle_completions(int client_fd, const std::string& body) {
        auto messages = json_get_messages(body);
        // 0 == "not set" — generate_fn / stream_generate_fn pick a sensible
        // default (run until end of context or natural EOS). Don't clip to
        // 500 here: the API caller may legitimately want a long response
        // and the engine has the KV context room for it.
        int max_tokens = json_get_int(body, "max_tokens", 0);
        bool stream = body.find("\"stream\"") != std::string::npos &&
                      (body.find("\"stream\":true") != std::string::npos ||
                       body.find("\"stream\": true") != std::string::npos);

        // Apply per-request sampling params from OpenAI API
        if (sampling_params) {
            float temp = json_get_float(body, "temperature", -1.0f);
            if (temp >= 0.0f) sampling_params->temperature = temp;
            float top_p = json_get_float(body, "top_p", -1.0f);
            if (top_p >= 0.0f) sampling_params->top_p = top_p;
            float freq_pen = json_get_float(body, "frequency_penalty", -999.0f);
            if (freq_pen > -999.0f) sampling_params->freq_penalty = freq_pen;
            float pres_pen = json_get_float(body, "presence_penalty", -999.0f);
            if (pres_pen > -999.0f) sampling_params->pres_penalty = pres_pen;
            // OpenAI doesn't have repetition_penalty but vLLM/llama.cpp accept it
            float rep_pen = json_get_float(body, "repetition_penalty", -1.0f);
            if (rep_pen > 0.0f) sampling_params->rep_penalty = rep_pen;
        }

        if (messages.empty()) {
            send_response(client_fd, 400, "application/json",
                "{\"error\":{\"message\":\"No messages provided\"}}");
            return;
        }

        // Extract tools definition and inject into system message
        std::string tools_json = json_get_raw(body, "tools");

        std::vector<std::pair<std::string, std::string>> msg_pairs;
        for (size_t mi = 0; mi < messages.size(); mi++) {
            auto& m = messages[mi];
            if (m.role == "assistant" && !m.tool_calls_json.empty()) {
                // Convert tool_calls to gist template XML format
                std::string content;
                if (!m.content.empty()) content = m.content;
                content += tool_calls_to_xml(m.tool_calls_json);
                msg_pairs.push_back({"assistant", content});
            } else if (m.role == "tool") {
                // Convert tool responses: group consecutive tool messages under one user turn
                // (matches Qwen3-Coder.jinja: <|im_start|>user\n<tool_response>...\n<|im_end|>)
                std::string tool_content;
                while (mi < messages.size() && messages[mi].role == "tool") {
                    tool_content += "<tool_response>\n" + messages[mi].content + "\n</tool_response>\n";
                    if (mi + 1 < messages.size() && messages[mi + 1].role == "tool") mi++;
                    else break;
                }
                msg_pairs.push_back({"user", tool_content});
            } else if (m.role == "developer") {
                if (msg_pairs.empty() || msg_pairs[0].first != "system")
                    msg_pairs.insert(msg_pairs.begin(), {"system", m.content});
                else
                    msg_pairs.push_back({"user", m.content});
            } else {
                msg_pairs.push_back({m.role, m.content});
            }
        }

        // Inject tools into system message (matches gist patched Qwen3.5 template exactly)
        if (!tools_json.empty()) {
            // Per-line JSON with spaces (matching Jinja: tool | tojson per line)
            std::string tools_lines;
            auto tool_objs = json_array_objects(tools_json);
            for (auto& t : tool_objs) tools_lines += "\n" + json_normalize(t);

            std::string tool_prompt =
                "# Tools\n\nYou have access to the following functions:\n\n<tools>"
                + tools_lines +
                "\n</tools>"
                "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                "<tool_call>\n<function=example_function_name>\n"
                "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
                "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                "that can span\nmultiple lines\n</parameter>\n"
                "</function>\n</tool_call>\n\n"
                "<IMPORTANT>\nReminder:\n"
                "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
                "- Required parameters MUST be specified\n"
                "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
                "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
                "</IMPORTANT>";

            // Gist template: tools go FIRST in system, user system msg appended after
            bool has_system = false;
            for (auto& [role, content] : msg_pairs) {
                if (role == "system") {
                    content = tool_prompt + "\n\n" + content;
                    has_system = true;
                    break;
                }
            }
            if (!has_system) {
                msg_pairs.insert(msg_pairs.begin(), {"system", tool_prompt});
            }
        }

        std::vector<int> prompt_ids;
        if (chat_encode_fn) prompt_ids = chat_encode_fn(msg_pairs);

        // Debug: dump prompt details
        { FILE* f = fopen("/tmp/engine_prompt_debug.txt", "w");
          if (f) {
              for (size_t i = 0; i < msg_pairs.size(); i++)
                  fprintf(f, "--- msg[%zu] role='%s' ---\n%s\n", i, msg_pairs[i].first.c_str(), msg_pairs[i].second.c_str());
              fprintf(f, "--- prompt_ids (%zu tokens) ---\n", prompt_ids.size());
              for (size_t i = 0; i < prompt_ids.size(); i++) fprintf(f, "%d ", prompt_ids[i]);
              fprintf(f, "\n");
              fclose(f);
          }
        }

        printf("[API] %zu messages, %zu prompt tokens, max_gen=%d, stream=%d, has_tools=%s\n",
               messages.size(), prompt_ids.size(), max_tokens, stream,
               tools_json.empty() ? "no" : "yes");
        for (size_t i = 0; i < msg_pairs.size(); i++)
            printf("  msg[%zu] role='%s' len=%zu: %.120s\n", i, msg_pairs[i].first.c_str(),
                   msg_pairs[i].second.size(), msg_pairs[i].second.c_str());

        if (stream && stream_generate_fn) {
            // SSE streaming response
            std::string header = "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: keep-alive\r\n"
                "Access-Control-Allow-Origin: *\r\n\r\n";
            ::send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

            // First chunk: role
            send_sse(client_fd, "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\""
                + model_name + "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}");

            std::string stream_accum;  // accumulate full output for tool_call detection
            stream_generate_fn(prompt_ids, max_tokens, [&](const std::string& token, bool done) {
                if (done) {
                    // Parse accumulated output for tool calls
                    auto parsed_calls = parse_tool_calls(stream_accum);
                    if (!parsed_calls.empty()) {
                        for (size_t ci = 0; ci < parsed_calls.size(); ci++) {
                            auto& c = parsed_calls[ci];
                            send_sse(client_fd, "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\""
                                + model_name + "\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{"
                                "\"index\":" + std::to_string(ci) + ",\"id\":\"call_" + std::to_string(ci) + "\","
                                "\"type\":\"function\",\"function\":{\"name\":\"" + json_escape(c.name) + "\","
                                "\"arguments\":\"" + json_escape(c.arguments) + "\"}}]},\"finish_reason\":null}]}");
                        }
                        send_sse(client_fd, "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\""
                            + model_name + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}");
                    } else {
                        send_sse(client_fd, "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\""
                            + model_name + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}");
                    }
                    send_sse(client_fd, "[DONE]");
                } else {
                    stream_accum += token;
                    send_sse(client_fd, "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\""
                        + model_name + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
                        + json_escape(token) + "\"},\"finish_reason\":null}]}");
                }
            });
        } else {
            // Non-streaming response
            std::string generated_text = generate_fn(prompt_ids, max_tokens);

            // Parse tool calls from XML output
            auto parsed_calls = parse_tool_calls(generated_text);
            std::string finish_reason = parsed_calls.empty() ? "stop" : "tool_calls";

            // Clean content: strip <think> and <tool_call> blocks
            std::string clean_content = strip_think_block(generated_text);
            clean_content = strip_tool_calls(clean_content);

            std::ostringstream json;
            json << "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"model\":\""
                 << model_name << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"";

            if (!parsed_calls.empty()) {
                // Return structured tool_calls + cleaned content
                json << ",\"content\":" << (clean_content.empty() ? "null" : "\"" + json_escape(clean_content) + "\"");
                json << ",\"tool_calls\":" << tool_calls_to_api_json(parsed_calls);
            } else {
                json << ",\"content\":\"" << json_escape(clean_content) << "\"";
            }

            json << "},\"finish_reason\":\"" << finish_reason << "\"}],"
                 << "\"usage\":{\"prompt_tokens\":" << prompt_ids.size()
                 << ",\"completion_tokens\":0,\"total_tokens\":" << prompt_ids.size() << "}}";

            send_response(client_fd, 200, "application/json", json.str());
        }
    }

    void handle_client(int client_fd) {
        std::vector<char> buf_vec(131072);  // 128KB heap buffer
        char* buf = buf_vec.data();
        int buf_size = (int)buf_vec.size();
        int total = 0;
        // Read headers + body
        while (total < buf_size - 1) {
            int n = recv(client_fd, buf + total, buf_size - 1 - total, 0);
            if (n <= 0) break;
            total += n;
            buf[total] = 0;
            // Check if we have full request (headers + body)
            char* body_start = strstr(buf, "\r\n\r\n");
            if (body_start) {
                body_start += 4;
                // Check Content-Length
                char* cl = strcasestr(buf, "Content-Length:");
                if (cl) {
                    int content_len = atoi(cl + 15);
                    int body_received = total - (body_start - buf);
                    if (body_received >= content_len) break;
                } else {
                    break;
                }
            }
        }
        buf[total] = 0;

        // Parse request line
        std::string request(buf, total);
        std::string method, path;
        {
            size_t sp1 = request.find(' ');
            size_t sp2 = request.find(' ', sp1 + 1);
            if (sp1 != std::string::npos && sp2 != std::string::npos) {
                method = request.substr(0, sp1);
                path = request.substr(sp1 + 1, sp2 - sp1 - 1);
            }
        }

        // Extract body
        std::string body;
        size_t body_pos = request.find("\r\n\r\n");
        if (body_pos != std::string::npos)
            body = request.substr(body_pos + 4);

        // API key check
        if (!api_key.empty()) {
            std::string auth;
            size_t auth_pos = request.find("Authorization: Bearer ");
            if (auth_pos != std::string::npos) {
                size_t start = auth_pos + 22;
                size_t end = request.find("\r\n", start);
                auth = request.substr(start, end - start);
            }
            if (auth != api_key && method != "OPTIONS" && path != "/" && path != "/health") {
                send_response(client_fd, 401, "application/json",
                    "{\"error\":{\"message\":\"Invalid API key\"}}");
                close(client_fd);
                return;
            }
        }

        // CORS preflight
        if (method == "OPTIONS") {
            send_response(client_fd, 200, "text/plain", "");
        }
        // GET /v1/models
        else if (method == "GET" && (path == "/v1/models" || path == "/api/tags")) {
            handle_models(client_fd);
        }
        // POST /v1/chat/completions
        else if (method == "POST" && (path == "/v1/chat/completions" || path == "/api/chat")) {
            handle_completions(client_fd, body);
        }
        // Health check
        else if (method == "GET" && (path == "/" || path == "/health")) {
            send_response(client_fd, 200, "application/json",
                "{\"status\":\"ok\",\"model\":\"" + model_name + "\"}");
        }
        else {
            send_response(client_fd, 404, "application/json",
                "{\"error\":{\"message\":\"Not found\"}}");
        }

        close(client_fd);
    }

    bool start(int _port) {
        port = _port;
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) { perror("socket"); return false; }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;  // bind to all interfaces
        addr.sin_port = htons(port);

        if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind"); return false;
        }
        if (listen(server_fd, 16) < 0) {
            perror("listen"); return false;
        }

        printf("\n=== API Server listening on http://0.0.0.0:%d ===\n", port);
        printf("  POST /v1/chat/completions  (OpenAI compatible)\n");
        printf("  GET  /v1/models\n\n");

        while (true) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) { perror("accept"); continue; }

            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
            printf("[API] Connection from %s\n", client_ip);

            // Handle synchronously (one request at a time for GPU safety)
            handle_client(client_fd);
        }
        return true;
    }

    void stop() {
        if (server_fd >= 0) { close(server_fd); server_fd = -1; }
    }
};
