"""
提示词模板

引用格式要求来自 app/core/generate/citation.py —— 不许在这里另写一套,
否则又会出现"prompt 说什么、context 给什么"两边对不上的问题(T-01)。
"""
from __future__ import annotations

from typing import Dict

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


def build_system_prompt(context: str) -> Dict[str, str]:
    """构造系统消息。引用格式与上下文格式同源。"""
    if context and context.strip():
        return {
            "role": "system",
            "content": SYSTEM_PROMPT_WITH_CONTEXT.format(
                instruction=CITATION_INSTRUCTION,
                context=context,
            ),
        }
    return {"role": "system", "content": SYSTEM_PROMPT_NO_CONTEXT}
