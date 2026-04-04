import base64
import json
import uuid

import requests
import streamlit as st

# API 配置
BASE_URL = "http://127.0.0.1:8000/api"
REVIEW_API_URL = f"{BASE_URL}/review"
UPLOAD_KNOWLEDGE_API = f"{BASE_URL}/system/knowledge/upload"
DEBUG_RAG_API = f"{BASE_URL}/system/knowledge/debug"
CHAT_RAG_API = f"{BASE_URL}/system/knowledge/chat"
CHAT_GRAPH_API = f"{BASE_URL}/system/knowledge/chat/graph"

# ==========================================
# 🎨 全局样式与页面配置
# ==========================================
st.set_page_config(page_title="企业 AI 标书审查平台", page_icon="✨", layout="wide")

# 注入自定义 CSS 以美化界面（隐藏默认汉堡菜单、底部水印，调整边距）
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero-title {text-align: center; color: #1E88E5; font-size: 3rem; font-weight: 700; margin-top: 8vh;}
    .hero-subtitle {text-align: center; color: #607D8B; font-size: 1.2rem; margin-bottom: 10vh;}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 📍 左侧边栏 (导航与全局设置)
# ==========================================
with st.sidebar:
    st.title("✨ 标书审查 AI")
    st.caption("企业级多智能体协同平台")
    st.divider()

    # 使用 Radio 模拟现代 Web 应用的侧边栏导航
    nav_selection = st.radio(
        "导航菜单",
        [
            "💬 知识库问答 (标准版)",
            "🤖 智能体对话 (LangGraph)",
            "📝 智能标书审查",
            "📚 知识库管理",
            "🔍 RAG 穿透测试",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # 将 RAG 开关提升为全局侧边栏设置，保持对话界面纯净
    st.subheader("⚙️ 引擎设置")
    use_rag = st.toggle(
        "📚 开启知识库检索",
        value=True,
        help="开启后，对话和审查将严格基于企业规章库。关闭则使用大模型自由发散记忆。",
    )

    st.markdown("<div style='height:20vh;'></div>", unsafe_allow_html=True)
    st.caption("v2.0.1 | 纯异步多模态架构")


# ==========================================
# 3. 主界面 1 & 1.5: 融合版的对话模块
# ==========================================
if nav_selection in ["💬 知识库问答 (标准版)", "🤖 智能体对话 (LangGraph)"]:
    is_graph_mode = "LangGraph" in nav_selection
    current_api_url = CHAT_GRAPH_API if is_graph_mode else CHAT_RAG_API
    msg_key = "messages_graph" if is_graph_mode else "messages_std"

    # 🌟 新增：为当前模式分配一个独立的 Session ID 用于后端记忆隔离
    session_id_key = f"{msg_key}_session_id"
    if session_id_key not in st.session_state:
        st.session_state[session_id_key] = str(uuid.uuid4())

    # 🌟 状态隔离：为两个模式使用不同的历史记录 Key，避免切换时串台
    msg_key = "messages_graph" if is_graph_mode else "messages_std"

    if msg_key not in st.session_state:
        st.session_state[msg_key] = []

    # 1. 欢迎空状态
    if not st.session_state[msg_key]:
        title = "LangGraph 智能体引擎" if is_graph_mode else "标书审查 AI"
        st.markdown(
            f'<div class="hero-title">您好，我是{title}</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="hero-subtitle">我可以帮您解答招投标规章、合规红线以及历史标书中的任何问题。</div>',
            unsafe_allow_html=True,
        )
        # ...省略三个 col 的快捷提示语...

    # 2. 渲染历史对话 (注意改为遍历 st.session_state[msg_key])
    for message in st.session_state[msg_key]:
        avatar = (
            "🧑‍💻" if message["role"] == "user" else ("🤖" if is_graph_mode else "✨")
        )
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 参考的知识源"):
                    for idx, source in enumerate(message["sources"]):
                        st.caption(
                            f"来源 {idx + 1} | 相关度: {source.get('score', 'N/A')}"
                        )
                        st.write(source.get("text", ""))
                        st.divider()

    # 3. 底部吸附的输入框
    if prompt := st.chat_input("请输入您的问题，按回车发送..."):
        # 用户消息上屏
        st.session_state[msg_key].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # AI 思考与回复
        ai_avatar = "🤖" if is_graph_mode else "✨"
        with st.chat_message("assistant", avatar=ai_avatar):
            message_placeholder = st.empty()

            engine_name = "LangGraph 节点" if is_graph_mode else "底层系统"
            initial_status = (
                f"🔄 {engine_name}正在进行向量召回与重排..."
                if use_rag
                else f"🔄 {engine_name}正在思考..."
            )
            message_placeholder.markdown(f"*{initial_status}*")

            try:
                payload = {
                    "query": prompt,
                    "use_rag": use_rag,
                    "session_id": st.session_state[session_id_key],
                }
                # 发送到动态选择的 API
                response = requests.post(current_api_url, json=payload, stream=True)
                response.raise_for_status()

                full_answer = ""
                sources = []

                # 监听流式事件（标准版和 Graph 版返回的数据结构在经过我们封装后是完全一致的）
                for line in response.iter_lines():
                    if line:
                        event = json.loads(line.decode("utf-8"))

                        event_type = event.get("type")
                        if event_type == "sources":
                            sources = event.get("data", [])
                            message_placeholder.markdown("*✨ 正在生成回答...*")

                        elif event_type == "chunk":
                            full_answer += event.get("content", "")
                            message_placeholder.markdown(full_answer + "▌")

                        elif event_type == "error":
                            full_answer = f"⚠️ 发生错误: {event.get('message')}"
                            message_placeholder.error(full_answer)

                message_placeholder.markdown(full_answer)

                if sources:
                    with st.expander("📚 参考的知识源"):
                        for idx, source in enumerate(sources):
                            st.caption(
                                f"来源 {idx + 1} | 相关度: {source.get('score', 'N/A')}"
                            )
                            st.write(source.get("text", ""))
                            st.divider()

                # 存入对应模式的历史
                st.session_state[msg_key].append(
                    {"role": "assistant", "content": full_answer, "sources": sources}
                )

            except Exception as e:
                st.error(f"网络请求失败：{e}")

# ==========================================
# 主界面 2: 📝 智能标书审查 (Agent 模式)
# ==========================================

elif nav_selection == "📝 智能标书审查":
    st.header("📝 标书智能化审查引擎")
    st.markdown(
        "上传标书文本与资质图片，由多 Agent 智能体并发执行**全文精读**与**双向合规对比**。"
    )
    st.divider()

    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.subheader("📤 提交材料")
        doc_text = st.text_area(
            "粘贴标书核心文本段落：", height=220, placeholder="在此输入标书文本..."
        )
        uploaded_images = st.file_uploader(
            "上传资质图片或营业执照",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        # 开放高级策略开关给用户
        with st.expander("⚙️ 审查策略设置"):
            enable_full_text = st.checkbox("开启逐段精读 (耗时较长)", value=True)
            enable_double_rag = st.checkbox("开启双向 RAG 核对", value=True)

        submit_btn = st.button(
            "🚀 启动多智能体审查", type="primary", use_container_width=True
        )

    with col2:
        st.subheader("📊 审查报告")
        if submit_btn and doc_text.strip():
            with st.status("🤖 正在启动 AI 审查流水线...", expanded=True) as status:
                log_container = st.container()
                # 专门为刷屏的 Chunk 进度准备一个“原地刷新”的占位符
                chunk_progress_placeholder = st.empty()

                try:
                    payload = {
                        "document_text": doc_text,
                        "images_base64": [
                            base64.b64encode(img.read()).decode("utf-8")
                            for img in (uploaded_images or [])
                        ]
                        if uploaded_images
                        else [],
                        "enable_full_text_check": enable_full_text,
                        "enable_double_rag_check": enable_double_rag,
                    }

                    # 发起流式请求
                    with requests.post(
                        f"{REVIEW_API_URL}/stream", json=payload, stream=True
                    ) as response:
                        for line in response.iter_lines():
                            if line:
                                event = json.loads(line.decode("utf-8"))

                                if event["status"] == "processing":
                                    # UX 优化：如果是全文通读的高频日志，原地覆盖刷新
                                    if "全文通读" in event["agent"]:
                                        chunk_progress_placeholder.info(
                                            f"⏳ **{event['agent']}**: {event['message']}"
                                        )
                                    else:
                                        log_container.write(
                                            f"⏳ **{event['agent']}**: {event['message']}"
                                        )

                                elif event["status"] == "start":
                                    log_container.write(f"🌟 {event['message']}")

                                elif event["status"] == "final":
                                    review_result = event["data"]
                                    status.update(
                                        label="✨ 审查圆满完成",
                                        state="complete",
                                        expanded=False,
                                    )

                                    # 清空占位符
                                    chunk_progress_placeholder.empty()

                                    st.divider()
                                    if review_result["has_issues"]:
                                        st.error("⚠️ 发现潜在风险点，详情如下：")
                                        for issue in review_result["issues"]:
                                            # 根据分类展示不同颜色的提示框
                                            cat = issue.get("category", "")
                                            msg = issue.get("message", "")
                                            evidence = issue.get("evidence", "")
                                            rule = issue.get("reference_rule", "")

                                            with st.expander(
                                                f"🚩 [{cat}] {msg[:30]}..."
                                            ):
                                                st.warning(f"**问题描述**: {msg}")
                                                if evidence:
                                                    st.info(f"**标书原文**: {evidence}")
                                                if rule:
                                                    st.error(f"**违反规章**: {rule}")
                                    else:
                                        st.success(
                                            "🎉 完美！审查通过，未发现任何违规。"
                                        )

                                elif event["status"] == "error":
                                    status.update(label="❌ 审查中断", state="error")
                                    st.error(event["message"])

                except Exception as e:
                    status.update(label="❌ 连接失败", state="error")
                    st.error(f"连接审查引擎失败: {e}")

# ==========================================
# 主界面 3: 📚 知识库管理
# ==========================================

elif nav_selection == "📚 知识库管理":
    st.header("📚 企业知识中枢管理")
    st.markdown(
        "上传企业的规章制度、历史高分标书等资料，系统将自动利用大模型进行语义切片并注入底层的 Milvus 向量库。"
    )
    st.divider()

    knowledge_files = st.file_uploader(
        "拖拽或选择 PDF / TXT / MD 文档",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
    )

    if st.button("🔄 上传并重建混合检索向量库", type="primary"):
        if knowledge_files:
            with st.spinner(
                "📦 正在上传并触发后台异步向量化流水线 (Ingestion Pipeline)..."
            ):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), "application/octet-stream"))
                        for f in knowledge_files
                    ]
                    response = requests.post(UPLOAD_KNOWLEDGE_API, files=files_payload)
                    response.raise_for_status()
                    st.success(f"✅ {response.json()['message']}")
                    st.balloons()
                except Exception as e:
                    st.error(f"上传失败：{e}")
        else:
            st.warning("请先选择要上传的文件！")

# ==========================================
# 主界面 4: 🔍 RAG 穿透测试
# ==========================================

elif nav_selection == "🔍 RAG 穿透测试":
    st.header("🔍 RAG 检索管道穿透测试 (Debug 工具)")
    st.markdown(
        "跳过大模型的回答阶段，直击底层数据库，查看 **Milvus 双路召回** 和 **Rerank 交叉重排** 的原始打分明细。"
    )
    st.divider()

    test_query = st.text_input("输入检索测试词 (例如：响应时间红线)", value="响应时间")

    if st.button("🧪 发起底层探测", icon="🩻", type="primary"):
        if test_query:
            with st.spinner("正在穿透 LlamaIndex 抓取原始 Node 数据..."):
                try:
                    response = requests.post(DEBUG_RAG_API, json={"query": test_query})
                    response.raise_for_status()
                    data = response.json()
                    st.success(
                        f"🎯 初筛召回 {data['raw_recall_count']} 条数据，重排淘汰后精选 {data['reranked_count']} 条。"
                    )

                    col_raw, col_rerank = st.columns([1, 1], gap="large")

                    with col_raw:
                        st.subheader("🌪️ 粗排混合召回库")
                        st.caption("Milvus 底层 Sparse+Dense 融合结果")
                        for idx, node in enumerate(data["raw_nodes"]):
                            with st.container(border=True):
                                st.write(f"**#{idx + 1} | 分数: `{node['score']}`**")
                                st.caption(f"来源: {node['file_name']}")
                                st.write(node["text"])

                    with col_rerank:
                        st.subheader("🎯 Rerank 精排库")
                        st.caption("私有化 Cross-Encoder 最终输送给大模型的片段")
                        for idx, node in enumerate(data["reranked_nodes"]):
                            with st.container(border=True):
                                st.info(
                                    f"**🏅 精选 #{idx + 1} | 精确度: `{node['score']}`**\n\n{node['text']}"
                                )

                except Exception as e:
                    st.error(f"探测请求失败：{e}")
