import base64

import requests
import streamlit as st

# API 配置
BASE_URL = "http://127.0.0.1:8000/api"
REVIEW_API_URL = f"{BASE_URL}/review"
UPLOAD_KNOWLEDGE_API = f"{BASE_URL}/system/knowledge/upload"
DEBUG_RAG_API = f"{BASE_URL}/system/knowledge/debug"
CHAT_RAG_API = f"{BASE_URL}/system/knowledge/chat"

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
        ["💬 知识库问答", "📝 智能标书审查", "📚 知识库管理", "🔍 RAG 穿透测试"],
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
# 主界面 1: 💬 知识库问答 (Gemini 风格)
# ==========================================
if nav_selection == "💬 知识库问答":
    # 状态初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 1. 欢迎空状态 (模仿 Gemini)
    if not st.session_state.messages:
        st.markdown(
            '<div class="hero-title">您好，我是标书审查 AI</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-subtitle">我可以帮您解答招投标规章、合规红线以及历史标书中的任何问题。</div>',
            unsafe_allow_html=True,
        )

        # 提供快捷提示词
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("💡 **合规提问**\n\n我们公司的系统响应时间红线是多少？")
        with col2:
            st.info("💡 **流程咨询**\n\n标书资质文件必须要盖鲜章吗？")
        with col3:
            st.info("💡 **历史检索**\n\n提取上一版移动项目标书的核心报价。")

    # 2. 渲染历史对话
    for message in st.session_state.messages:
        # 自定义头像：User 用人像，AI 用闪烁星（Gemini 标志性风格）
        avatar = "🧑‍💻" if message["role"] == "user" else "✨"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            # 渲染历史来源
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # AI 思考与回复
        with st.chat_message("assistant", avatar="✨"):
            loading_text = (
                "✨ 正在检索知识库并思考回答..."
                if use_rag
                else "✨ 正在进行独立逻辑推理..."
            )
            with st.spinner(loading_text):
                try:
                    payload = {"query": prompt, "use_rag": use_rag}
                    response = requests.post(CHAT_RAG_API, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    answer = data.get("answer", "未能获取有效回答。")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 参考的知识源"):
                            for idx, source in enumerate(sources):
                                st.caption(
                                    f"来源 {idx + 1} | 相关度: {source.get('score', 'N/A')}"
                                )
                                st.write(source.get("text", ""))
                                st.divider()

                    # 存入历史（同时保存 sources）
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except Exception as e:
                    st.error(f"网络请求失败：{e}")


# ==========================================
# 主界面 2: 📝 智能标书审查 (Agent 模式)
# ==========================================

elif nav_selection == "📝 智能标书审查":
    st.header("📝 标书智能化审查引擎")
    st.markdown(
        "上传标书文本与资质图片，由多 Agent 智能体并发执行**语法、逻辑、合规、视觉**四大维度的严格审查。"
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

        submit_btn = st.button(
            "🚀 启动多智能体审查", type="primary", use_container_width=True
        )

    with col2:
        st.subheader("📊 审查报告")
        if submit_btn and doc_text.strip():
            with st.status("🤖 Agent 正在并发作业中...", expanded=True) as status:
                st.write("✅ 语法纠错智能体已启动")
                st.write("✅ 关键信息抽取智能体已启动")
                st.write("✅ 知识库合规校验已启动")
                if uploaded_images:
                    st.write("✅ 多模态视觉审计已启动")

                images_b64 = [
                    base64.b64encode(img.read()).decode("utf-8")
                    for img in (uploaded_images or [])
                ]
                try:
                    res = requests.post(
                        REVIEW_API_URL,
                        json={"document_text": doc_text, "images_base64": images_b64},
                    )
                    res.raise_for_status()
                    result = res.json()
                    status.update(label="审查完成", state="complete", expanded=False)

                    # 渲染结果
                    if result["has_issues"]:
                        st.error("⚠️ 标书存在潜在风险点！")
                        for issue in result["issues"]:
                            # 根据问题类型使用不同的提示框
                            if "合规" in issue["category"]:
                                st.error(
                                    f"**[{issue['category']}]** {issue['message']}"
                                )
                            elif "语法" in issue["category"]:
                                st.warning(
                                    f"**[{issue['category']}]** {issue['message']}"
                                )
                            else:
                                st.info(f"**[{issue['category']}]** {issue['message']}")
                    else:
                        st.success("🎉 完美！审查通过，未发现任何违规。")

                    if result.get("extracted_data"):
                        st.divider()
                        st.write("📌 **AI 提取的关键指标**")
                        st.json(result["extracted_data"])
                except Exception as e:
                    status.update(label="审查失败", state="error")
                    st.error(f"API 调用失败：{e}")

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
