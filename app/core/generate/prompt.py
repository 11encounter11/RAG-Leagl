"""
提示词模板

引用格式要求来自 app/core/generate/citation.py —— 不许在这里另写一套,
否则又会出现"prompt 说什么、context 给什么"两边对不上的问题(T-01)。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.core.generate.citation import CITATION_INSTRUCTION

SYSTEM_PROMPT_WITH_CONTEXT = """你是一个严谨的法律信息助手，基于知识库中的资料回答用户问题。

回答要求：
{instruction}

参考资料：
{context}"""

SYSTEM_PROMPT_NO_CONTEXT = """你是一个严谨的法律信息助手。

当前没有检索到可用的参考资料，因此：
1. 不要引用任何法条编号，也不要给出具体法律结论；
2. 用一两句话说明你没有找到相关依据，并请用户补充更具体的问题
   （例如涉及什么类型的纠纷、在哪个城市、事情经过）；
3. 不要编造法条，也不要假装引用。"""


def build_system_prompt(
    context: str, missing: Optional[List[str]] = None
) -> Dict[str, str]:
    """构造系统消息。引用格式与上下文格式同源。

    `missing` 是反思节点给出的"资料未覆盖项"(P2-04)。检索只能按相关性取前 K 条,
    它不会自动补全一个法律论证所需的例外/条件; 当反思已经指出缺口时, 必须让
    生成模型知道, 否则它会拿半套依据给出一个听起来很确定的结论。
    """
    if context and context.strip():
        content = SYSTEM_PROMPT_WITH_CONTEXT.format(
            instruction=CITATION_INSTRUCTION,
            context=context,
        )
        gaps = [str(item).strip() for item in (missing or []) if str(item).strip()]
        if gaps:
            content += (
                "\n\n检索完整性提示：\n"
                "现有参考资料未覆盖以下内容：\n"
                + "\n".join(f"- {gap}" for gap in gaps)
                + "\n回答时必须明确说明这些部分未能核实，"
                "不得给出超出资料的绝对结论。"
            )
        return {"role": "system", "content": content}
    return {"role": "system", "content": SYSTEM_PROMPT_NO_CONTEXT}
