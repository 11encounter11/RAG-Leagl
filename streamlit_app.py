"""
Streamlit 前端页面 - 法律智能问答系统
"""
import streamlit as st
import requests
import json
import time

# ===== 页面配置 =====
st.set_page_config(
    page_title="法律智能问答系统",
    page_icon="️",
    layout="wide",
)

# ===== 配置 =====
API_BASE = "http://localhost:8000"


# ===== 侧边栏 =====
with st.sidebar:
    st.title("⚖️ 法律智能问答")

    st.markdown("---")

    # 模型选择
    st.subheader("模型选择")
    model_choice = st.radio(
        "选择 LLM 模型",
        ["glm-5.3-flash", "deepseek-v4.1-flash"],
        format_func=lambda x: {
            "glm-5.3-flash": "智谱 GLM-5.3-Flash",
            "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
        }[x],
    )

    st.markdown("---")

    # 检索设置
    st.subheader("检索设置")
    top_k = st.slider("Top-K（返回片段数）", 1, 10, 5)
    similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")

    # 知识库管理
    st.subheader("知识库")
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "md", "txt"],
    )
    if uploaded_file:
        with st.spinner("正在上传并索引文档..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{API_BASE}/api/knowledge/upload", files=files)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f" 上传失败: {resp.json().get('detail', '未知错误')}")
            except requests.ConnectionError:
                st.error("❌ 无法连接后端服务，请先启动 FastAPI")

    st.markdown("---")

    # 会话管理
    st.subheader("会话历史")
    try:
        convs_resp = requests.get(f"{API_BASE}/api/conversations")
        if convs_resp.status_code == 200:
            convs = convs_resp.json()
            if convs:
                for conv in convs[:10]:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(f"💬 {conv['title']}")
                        st.caption(f"   {conv['model_name']} | {conv['message_count']}条消息")
                    with col2:
                        if st.button("", key=f"del_{conv['id']}"):
                            requests.delete(f"{API_BASE}/api/conversations/{conv['id']}")
                            st.rerun()
            else:
                st.caption("暂无会话")
    except requests.ConnectionError:
        st.caption("无法加载会话列表")


# ===== 主聊天区域 =====
st.header("⚖️ 法律智能问答系统")
st.caption("基于 RAG 检索增强生成，支持多模型切换")

# 初始化会话状态
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入
if prompt := st.chat_input("请输入你的法律问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用后端 API（流式）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            payload = {
                "message": prompt,
                "model_name": model_choice,
                "conversation_id": st.session_state.conversation_id,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
            }

            response = requests.post(
                f"{API_BASE}/api/chat/stream",
                json=payload,
                stream=True,
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if data.get("type") == "token":
                                full_response += data["content"]
                                message_placeholder.markdown(full_response + "▌")
                                # 保存 conversation_id
                                if data.get("conversation_id"):
                                    st.session_state.conversation_id = data["conversation_id"]
                            elif data.get("type") == "done":
                                break
                            elif data.get("type") == "error":
                                full_response = f"❌ 错误: {data['content']}"
                                break
            else:
                full_response = f"❌ 请求失败: {response.text}"

        except requests.ConnectionError:
            full_response = "❌ 无法连接后端服务，请先启动 FastAPI（`python -m app.main`）"
        except Exception as e:
            full_response = f"❌ 发生错误: {str(e)}"

        message_placeholder.markdown(full_response)

    # 保存助手消息
    st.session_state.messages.append({"role": "assistant", "content": full_response})
