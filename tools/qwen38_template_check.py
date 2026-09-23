#!/usr/bin/env python3
"""Diff the server's Qwen3.8 prompt against the official chat template.

Builds tools/test_qwen38_template.cpp (the server's render_qwen38_chat path),
feeds it a set of OpenAI-style chat requests, renders the same requests with
Qwen3.8's chat_template.jinja (vendored in tools/testdata, taken from
Qwen/Qwen3.8-27B) the way transformers / vLLM do, and compares the text.
No GPU or model needed; requires jinja2 and g++ plus the CUDA headers
(sampling.h includes cuda_fp16.h).

    python3 tools/qwen38_template_check.py [--cuda-include DIR ...] [-v]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "testdata", "qwen38_chat_template.jinja")

# Must match qwen38_effort_level() in src/server.h.
EFFORT_LEVEL = {"xhigh": "xhigh", "high": "xhigh", "max": "xhigh", "medium": "medium",
                "low": "low", "minimal": "low", "none": "none"}

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search", "description": "Search the web",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}


def call(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


AGENT_LOOP = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Seoul and Busan?"},
    {"role": "assistant", "content": None, "reasoning_content": "Need both cities.",
     "tool_calls": [call("c1", "get_weather", {"city": "Seoul"}),
                    call("c2", "get_weather", {"city": "Busan", "days": 3})]},
    {"role": "tool", "tool_call_id": "c1", "content": "Seoul: 21C, clear"},
    {"role": "tool", "tool_call_id": "c2", "content": "Busan: 24C, rain\n"},
    {"role": "assistant", "content": "Seoul is clear at 21C; Busan is rainy at 24C.",
     "reasoning_content": "Summarise both."},
    {"role": "user", "content": "Thanks. Search for umbrellas?"},
]

MULTI_TURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4", "reasoning_content": "\n2 plus 2 is 4.\n"},
    {"role": "user", "content": "  And times 3?  "},
]

# (name, request body, force_think handed over by the directive logic)
CASES = [
    ("single user, defaults", {"messages": [{"role": "user", "content": "hi"}]}, 0),
    ("system + user", {"messages": [{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "hello"}]}, 0),
    ("empty system", {"messages": [{"role": "system", "content": "  "},
                                   {"role": "user", "content": "hello"}]}, 0),
    ("thinking off (enable_thinking false)",
     {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
      "chat_template_kwargs": {"enable_thinking": False}}, -1),
    ("thinking on (enable_thinking true)",
     {"messages": [{"role": "user", "content": "hi"}],
      "chat_template_kwargs": {"enable_thinking": True}}, 1),
    ("effort low (top-level)", {"messages": [{"role": "user", "content": "hi"}],
                                "reasoning_effort": "low"}, 0),
    ("effort medium (kwargs)", {"messages": [{"role": "user", "content": "hi"}],
                                "chat_template_kwargs": {"reasoning_effort": "medium"}}, 0),
    ("effort low (extra_body kwargs)",
     {"messages": [{"role": "user", "content": "hi"}],
      "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}}}, 0),
    ("effort high -> xhigh", {"messages": [{"role": "user", "content": "hi"}],
                              "reasoning_effort": "high"}, 0),
    ("effort none -> thinking off", {"messages": [{"role": "user", "content": "hi"}],
                                     "reasoning_effort": "none"}, 0),
    ("multi-turn, preserve_thinking default", {"messages": MULTI_TURN}, 0),
    ("multi-turn, preserve_thinking false",
     {"messages": MULTI_TURN, "chat_template_kwargs": {"preserve_thinking": False}}, 0),
    ("multi-turn, reasoning field alias",
     {"messages": [MULTI_TURN[0], {"role": "assistant", "content": "4", "reasoning": "sum"},
                   MULTI_TURN[2]]}, 0),
    ("agent loop with tools", {"messages": AGENT_LOOP, "tools": [WEATHER_TOOL, SEARCH_TOOL]}, 0),
    ("agent loop, preserve_thinking false",
     {"messages": AGENT_LOOP, "tools": [WEATHER_TOOL, SEARCH_TOOL],
      "chat_template_kwargs": {"preserve_thinking": False}}, 0),
    ("agent loop mid-turn, preserve false (turns after the query keep think)",
     {"messages": AGENT_LOOP[:5], "tools": [WEATHER_TOOL],
      "chat_template_kwargs": {"preserve_thinking": False}}, 0),
    ("tools without system, thinking off",
     {"messages": [{"role": "user", "content": "weather?"}], "tools": [WEATHER_TOOL],
      "chat_template_kwargs": {"enable_thinking": False}}, -1),
    ("assistant text + tool call, non-string args",
     {"messages": [
         {"role": "user", "content": "go"},
         {"role": "assistant", "content": "Let me check.",
          "tool_calls": [call("c1", "search", {"q": "x", "n": 5, "exact": True, "ratio": 0.5,
                                               "filters": {"lang": "ko", "tags": ["a", "b"]},
                                               "none": None})]},
         {"role": "tool", "content": "[]"},
     ], "tools": [SEARCH_TOOL]}, 0),
    ("tool call with empty arguments",
     {"messages": [{"role": "user", "content": "go"},
                   {"role": "assistant", "content": "", "tool_calls": [call("c1", "search", {})]},
                   {"role": "tool", "content": "ok"}], "tools": [SEARCH_TOOL]}, 0),
    ("unicode text", {"messages": [{"role": "system", "content": "한국어로 답하세요."},
                                   {"role": "user", "content": "날씨 어때? 🌧"}]}, 0),
    ("unicode tool schema",
     {"messages": [{"role": "user", "content": "검색"}],
      "tools": [{"type": "function", "function": {"name": "검색", "description": "웹 검색",
                                                   "parameters": {"type": "object"}}}]}, 0),
]

# Written as raw JSON to exercise the parser on shapes json.dumps doesn't make.
RAW_CASES = [
    ("pretty-printed tools",
     '{\n  "messages": [ {"role": "user", "content": "weather?"} ],\n'
     '  "tools": [\n    {\n      "type": "function",\n      "function": {\n'
     '        "name": "get_weather",\n        "parameters": { "type": "object",\n'
     '          "properties": { "city": { "type": "string" } } }\n      }\n    }\n  ]\n}', 0),
    ("content null before tool_calls key",
     '{"messages":[{"role":"user","content":"go"},'
     '{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function",'
     '"function":{"name":"search","arguments":"{\\"q\\": \\"x\\"}"}}]},'
     '{"role":"tool","content":"done"}],"tools":[' + json.dumps(SEARCH_TOOL) + ']}', 0),
]


def jinja_render(body, force_think):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    with open(TEMPLATE, encoding="utf-8") as f:
        tmpl = env.from_string(f.read())

    # What vLLM hands the template: arguments decoded to a dict, `reasoning`
    # as reasoning_content.
    messages = []
    for m in body["messages"]:
        m = dict(m)
        if "reasoning" in m and "reasoning_content" not in m:
            m["reasoning_content"] = m.pop("reasoning")
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                args = tc["function"].get("arguments")
                if isinstance(args, str):
                    tc["function"]["arguments"] = json.loads(args) if args else {}
                tcs.append(tc)
            m["tool_calls"] = tcs
        messages.append(m)

    kwargs = {}
    for src in (body.get("extra_body", {}).get("chat_template_kwargs", {}),
                body.get("chat_template_kwargs", {})):
        kwargs.update(src)
    effort = kwargs.pop("reasoning_effort", None) or body.get("reasoning_effort")
    if effort:
        level = EFFORT_LEVEL[effort]
        if level == "none":
            kwargs["enable_thinking"] = False
        else:
            kwargs["reasoning_effort"] = level
    if force_think == -1:
        kwargs["enable_thinking"] = False
    return tmpl.render(messages=messages, tools=body.get("tools"),
                       add_generation_prompt=True, **kwargs)


def build(cuda_includes):
    out = os.path.join(tempfile.gettempdir(), "test_qwen38_template")
    cmd = ["g++", "-std=c++17", "-O0", "-Wall", "-Wno-unused-function",
           "-I" + os.path.join(HERE, "..", "src")]
    cmd += ["-I" + d for d in cuda_includes]
    cmd += ["-o", out, os.path.join(HERE, "test_qwen38_template.cpp")]
    subprocess.run(cmd, check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-include", action="append", default=[],
                    help="extra include dir for cuda_fp16.h (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    includes = a.cuda_include or ["/usr/local/cuda/include"]
    exe = build(includes)

    cases = [(n, json.dumps(b, ensure_ascii=False), b, ft) for n, b, ft in CASES]
    cases += [(n, raw, json.loads(raw), ft) for n, raw, ft in RAW_CASES]
    failed = 0
    with tempfile.TemporaryDirectory() as td:
        for name, raw, body, ft in cases:
            path = os.path.join(td, "req.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(raw)
            got = subprocess.run([exe, path, str(ft)], capture_output=True,
                                 check=True).stdout.decode("utf-8")
            want = jinja_render(body, ft)
            if got == want:
                print(f"  PASS  {name}")
                if a.verbose:
                    print(got)
                continue
            failed += 1
            i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]),
                     min(len(got), len(want)))
            print(f"  FAIL  {name}: first difference at char {i}")
            print(f"        engine: {got[max(0, i - 60):i + 60]!r}")
            print(f"        jinja : {want[max(0, i - 60):i + 60]!r}")
    print(f"{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
