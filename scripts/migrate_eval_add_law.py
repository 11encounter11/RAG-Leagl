"""
一次性迁移: 给评测集补 `law` 字段(已于 2026-10-14 执行完毕, 保留作记录)。

背景: 评测集原本只标条文号(如 `"第三十六条"`)。这在语料只有一部法时没问题,
但在语料已有 9 部法之后会产生**跨法误判** —— 劳动法第三十六条(工时上限)与
劳动合同法第三十六条(协商解除)会被判成同一条, 指标虚高且不报错。

本脚本给每条"可回答"样本补上 `law`。取值依据: 全部 36 条可回答样本的
标准答案条文号都属于《中华人民共和国劳动法》—— 这正是原语料(单文档)的由来。
超纲样本(正确答案是拒答)不标 `law`。

用法(幂等, 重复跑不会加重复字段):
    python scripts/migrate_eval_add_law.py --dry-run
    python scripts/migrate_eval_add_law.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET = Path("data/eval/legal_qa_eval.jsonl")
DEFAULT_LAW = "中华人民共和国劳动法"

# 字段顺序, 保证迁移后每行结构一致、diff 干净
FIELD_ORDER = ["id", "question", "kind", "answerable", "law", "required", "acceptable", "notes"]


def main() -> int:
    parser = argparse.ArgumentParser(description="给评测集补 law 字段")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--law", default=DEFAULT_LAW, help="给可回答样本填的法名")
    args = parser.parse_args()

    lines = [l for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    out: list[str] = []
    added = 0
    skipped = 0

    for raw in lines:
        if raw.strip().startswith("#"):
            out.append(raw)
            continue
        payload = json.loads(raw)
        answerable = bool(payload.get("answerable", True))

        if answerable and not payload.get("law"):
            payload["law"] = args.law
            added += 1
        elif answerable:
            skipped += 1
        else:
            # 超纲题: 正确答案是拒答, 没有"应该命中哪部法"
            payload.pop("law", None)
            payload.pop("laws", None)
            skipped += 1

        ordered = {key: payload[key] for key in FIELD_ORDER if key in payload}
        ordered.update({k: v for k, v in payload.items() if k not in FIELD_ORDER})
        out.append(json.dumps(ordered, ensure_ascii=False))

    print(f"样本 {len(lines)} 条: 补 law {added} 条, 跳过 {skipped} 条")
    if args.dry_run:
        print("[dry-run] 未写入")
        return 0
    DATASET.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"已写入 {DATASET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
