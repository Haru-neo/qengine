// Prints the prompt the server builds for a chat request with the Qwen3.8
// template (render_qwen38_chat), for tools/qwen38_template_check.py to diff
// against the official jinja template. No GPU or model needed.
//
//   g++ -std=c++17 -I../src -o /tmp/test_qwen38_template test_qwen38_template.cpp
//   /tmp/test_qwen38_template request.json [force_think]
//
// force_think is what the directive / enable_thinking logic hands over
// (-1 off, 0 unset, 1 on). The resolved value goes to stderr.
#include "sampling.h"
#include "server.h"
#include <cstdio>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s request.json [force_think]\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1]);
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    std::stringstream ss; ss << f.rdbuf();
    const std::string body = ss.str();
    int force_think = argc > 2 ? atoi(argv[2]) : 0;

    // Same order as HttpServer::handle_completions.
    auto messages = json_get_messages(body);
    std::string tools_json = json_get_raw(body, "tools");
    Qwen38ChatOptions o = qwen38_options_for_request(body, force_think);
    std::string prompt = render_qwen38_chat(messages, tools_json, o);
    fwrite(prompt.data(), 1, prompt.size(), stdout);
    fprintf(stderr, "force_think=%d\n", force_think);
    return 0;
}
