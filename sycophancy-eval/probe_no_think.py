#!/usr/bin/env python3
"""
probe_no_think.py — find out HOW to disable thinking on your endpoint.

Sends one moderately hard prompt per convention and reports, for each:
  - reasoning_tokens (from usage.completion_tokens_details, if exposed)
  - whether a reasoning_content field came back
  - final content length / finish_reason

The winning convention is the one with NO reasoning tokens and a NON-EMPTY
content that finishes normally ("stop") instead of hitting the token cap
("length"). Then run the pipeline with:
  --no-thinking --no-think-via <winner>

Usage:
  export OPENAI_BASE_URL=... ; export OPENAI_API_KEY=...
  python3 probe_no_think.py [model]        # default model: deepseek-flash
"""

import os
import sys

from openai import OpenAI

MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek-flash"
# Hard enough to trigger long thinking, easy enough to finish quickly without it.
PROMPT = (
    "Compute the number of ordered pairs (a, b) of positive integers with "
    "a + b <= 60 such that a divides b. Put your final answer in \\boxed{}."
)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL") or None,
    timeout=300,
    max_retries=0,
)


def probe(name, extra_body=None, suffix=""):
    kw = {"model": MODEL,
          "messages": [{"role": "user", "content": PROMPT + suffix}],
          "max_tokens": 4096}
    if extra_body is not None:
        kw["extra_body"] = extra_body
    try:
        r = client.chat.completions.create(**kw)
        ch = r.choices[0]
        msg = ch.message
        usage = r.usage
        det = getattr(usage, "completion_tokens_details", None)
        rtok = getattr(det, "reasoning_tokens", None) if det else None
        rc = getattr(msg, "reasoning_content", None)
        content = msg.content or ""
        verdict = (
            "✅ 思考已关闭" if (not rc and not rtok and content and ch.finish_reason == "stop")
            else "❌ 仍在思考/被截断"
        )
        print(
            f"[{name}] {verdict}\n"
            f"    finish={ch.finish_reason}  completion_tokens={usage.completion_tokens}  "
            f"reasoning_tokens={rtok}  reasoning_content={'yes(' + str(len(rc)) + '字)' if rc else 'no'}  "
            f"content_len={len(content)}"
        )
    except Exception as e:
        print(f"[{name}] ERROR（该约定被端点拒绝）: {str(e)[:150]}")


probe("baseline 什么都不加")
probe("extra_body: enable_thinking + chat_template_kwargs",
      extra_body={"enable_thinking": False,
                  "chat_template_kwargs": {"enable_thinking": False}})
probe("effort: reasoning_effort=minimal",
      extra_body={"reasoning_effort": "minimal"})
probe("thinking_flag: thinking={type:disabled}  (DeepSeek官方约定)",
      extra_body={"thinking": {"type": "disabled"}})
probe("prompt_suffix: ' /no_think'", suffix=" /no_think")
