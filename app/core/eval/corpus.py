"""
语料索引 —— 把 data/raw 下的语料跑一遍解析, 得到"每部法实际有哪些条文"。

两个用途:
  1. 评测集自检: 校验标准答案的条文号确实存在于它所属的那部法里
  2. 指标口径: 语料含多部法后, 匹配必须带上法名(见 metrics.grade_hit)

为什么单独成一个模块:
  这段逻辑之前写在 scripts/run_eval.py 里, 而且**只读 .txt/.md** ——
  语料换成官方 .docx 之后它就静默地什么都没收集到, 自检等于空跑。
  放进模块 + 用生产同一条 loader/ingest 链路, 才能被测试覆盖。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = Path("data/raw")
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt"}


def build_corpus_index(corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> Dict[str, Set[str]]:
    """返回 {法名: {条文号, ...}}。

    刻意复用生产同一条 ingest 链路(loader -> ingest_text): 如果这里另写一套正则
    去数条文, 就变成"用一个可能同样有问题的实现去验证另一个实现", 结论没有意义。

    同一部法有多个版本时, 条文号取**并集** —— 自检关心的是"这个条文号在这部法里
    存在吗"; 版本时效的判定是检索层的事, 不在这里。
    """
    from app.core.ingest import ingest_text
    from app.core.rag.loader import DocumentLoader

    directory = Path(corpus_dir)
    index: Dict[str, Set[str]] = {}
    if not directory.is_dir():
        logger.warning("语料目录不存在: %s", directory)
        return index

    loader = DocumentLoader()
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            documents = loader.load_file(str(path))
        except Exception as exc:  # noqa: BLE001 - 单份语料读不了不该中断整体
            logger.warning("跳过 %s: %s", path.name, exc)
            continue

        text = "\n\n".join((doc.text or "") for doc in documents).strip()
        if not text:
            continue

        result, _, _ = ingest_text(text, filename=path.name)
        for section in result.sections:
            law_name = (section.extra or {}).get("law_name")
            if not law_name or not section.article_no:
                continue
            index.setdefault(str(law_name), set()).add(section.article_no)

    return index


def list_laws(corpus_index: Dict[str, Set[str]]) -> List[str]:
    """语料里出现过的法名(排序)。"""
    return sorted(corpus_index)
