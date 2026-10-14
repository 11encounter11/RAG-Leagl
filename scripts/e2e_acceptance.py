# -*- coding: utf-8 -*-
"""端到端验收脚本: T-20 第 1/3 项 + 拒答/警示 + 重排器调用计数"""
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import requests

API = "http://localhost:8000"


def ask(message, **kw):
    payload = {"message": message, "model_name": "deepseek-v4.1-flash"}
    payload.update(kw)
    r = requests.post(f"{API}/api/chat", json=payload, timeout=300)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:2000]}")
        sys.exit(2)
    return r.json()


def report(tag, d):
    print(f"\n{'='*70}\n[{tag}] question in response below")
    print("retrieval_count =", d.get("retrieval_count"))
    print("domain_assessment =", d.get("domain_assessment"))
    print("workflow_steps =", json.dumps(d.get("workflow_steps"), ensure_ascii=False)[:400])
    print("conversation_id =", d.get("conversation_id"))
    srcs = d.get("sources") or []
    print(f"sources ({len(srcs)}):")
    for s in srcs[:5]:
        print(f"  #{s.get('index')} {s.get('law_name')} {s.get('article_no')} "
              f"norm={s.get('score_normalized'):.4f} raw={s.get('score_raw'):.4f}")
    print("--- answer (first 900 chars) ---")
    print((d.get("content") or d.get("answer") or "")[:900])


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "劳动法规定的试用期最长可以多久？"
    tag = sys.argv[2] if len(sys.argv) > 2 else "in_scope"
    d = ask(q)
    report(tag, d)
    with open(f"e2e_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
