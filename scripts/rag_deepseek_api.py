import os
import time
import gradio
import asyncio
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from functools import lru_cache
import diskcache as dc
import hashlib

# ------------------- 1. 环境 & 路径 -------------------
load_dotenv()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
print(f"base_dir: {base_dir}")

OUTPUT_DIR = os.path.join(base_dir, os.getenv("OUTPUT_DIR", "output/"))
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

persist_directory = os.path.join(OUTPUT_DIR, "chroma_db")
api_key = os.getenv("DEEPSEEK_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------- 2. 创建/加载 Chroma 向量库 -------------------
if not os.path.exists(persist_directory):
    # 读取所有 chunk_*.txt 并带上 source 元数据
    chunks = []
    texts = []
    metadatas = []

    if os.path.isdir(OUTPUT_DIR):
        for fn in sorted(os.listdir(OUTPUT_DIR)):
            if fn.startswith("chunk_") and fn.endswith(".txt"):
                path = os.path.join(OUTPUT_DIR, fn)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    texts.append(content)
                    metadatas.append({"source": fn})  # 关键：保存文件名
        if not texts:
            raise FileNotFoundError(f"{OUTPUT_DIR} 中未找到 chunk_*.txt")
    else:
        raise FileNotFoundError(f"{OUTPUT_DIR} 不存在")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,  # 关键：传入元数据
        persist_directory=persist_directory
    )
    print("Chroma 数据库创建完成（已包含 source 元数据）")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Chroma 数据库已加载")


# 持久化磁盘缓存
CACHE_DIR = os.path.join(OUTPUT_DIR, "rag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
rag_cache = dc.Cache(CACHE_DIR, disk_min_file_size=0)  # 持久化磁盘缓存
print(f"RAG 缓存目录: {CACHE_DIR}")

# ------------------- 3. DeepSeek LLM -------------------
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY 未设置")

llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=500
)
print("DeepSeek LLM 配置成功")

# ------------------- 4. Prompt -------------------
prompt_template = """
你是课程学习助手。

按以下步骤推理：
1. 识别上下文中的相关信息。
2. 总结或推测答案。
3. 注明来源块编号（如 chunk_3）。

示例：
- 问题: 什么是敏捷开发？
- 上下文: chunk_10: Agile means iterative development...
- 答案: 敏捷开发是一种迭代和增量开发的软件开发方法。
- 来源: chunk_10

基于以下课程上下文，回答问题。优先提取关键信息，若信息不足，可基于相关概念推测或总结，并注明来源块编号。若完全无法回答，说“未知”。

上下文: {context}
问题: {question}
答案:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ------------------- 5. MMR 检索器（动态 k） -------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",                 # Maximum Marginal Relevance
    search_kwargs={"k": 15, "fetch_k": 30}   # k=15 最终返回，fetch_k=30 候选池
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)


def cached_rag(query: str) -> str:
    """
    持久化磁盘缓存 RAG 问答（跨进程生效）
    - 第一次运行：完整 RAG → 存入磁盘
    - 第二次运行：直接从磁盘读取 → 0.01s
    """
    # 标准化查询 + 生成唯一 key
    normalized = query.strip().lower()
    cache_key = f"rag_v1:{hashlib.md5(normalized.encode()).hexdigest()}"

    if cache_key in rag_cache:
        print(f"[缓存命中] {cache_key[-8:]}")
        return rag_cache[cache_key]

    print(f"[缓存未命中] 执行 RAG...")
    result = qa_chain.invoke({"query": query})["result"]
    rag_cache[cache_key] = result
    return result


# ------------------- 6. 测试 RAG（磁盘缓存 + 跨进程命中） -------------------
query = "组织结构有哪几种类型？请描述每种类型对项目管理的影响，并举例说明。"

try:
    # 1）检索调试（仅在缓存未命中时执行）

    retrieved_docs = retriever.invoke(query)
    print("\n=== 检索到的文档（MMR, k=15） ===")
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "")
        if source:
            chunk_id = os.path.splitext(source)[0]  # 得到 chunk_3
        else:
            chunk_id = f"unknown_{i}"
        preview = doc.page_content.replace("\n", " ")[:120]
        print(f"Doc {i} ({chunk_id}): {preview}...")


    start_total = time.time()

    # === 磁盘缓存 RAG 调用 ===
    answer = cached_rag(query)

    response_time = time.time() - start_total

    # 2）打印答案 + 响应时间
    print("\n" + "="*70)
    print("DeepSeek RAG 回答:")
    print(answer)
    print(f"总响应时间: {response_time:.3f} 秒")
    print("="*70)

    # 3）打印缓存状态
    normalized = query.strip().lower()
    cache_key = f"rag_v1:{hashlib.md5(normalized.encode()).hexdigest()}"
    print(f"缓存键: {cache_key[-12:]}")
    print(f"缓存目录: {CACHE_DIR}")
    print(f"当前缓存大小: {len(rag_cache)} 条")

    # 4）可选：打印缓存命中统计
    if cache_key in rag_cache:
        print("缓存状态: 命中（第二次运行脚本将直接读取）")
    else:
        print("缓存状态: 未命中（已写入磁盘，下次运行将命中）")

except Exception as e:
    print(f"问答测试失败: {str(e)}")
    raise

# ------------------- 7. Gradio UI（异步 + 计时） -------------------
async def ask_question(question: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        start = time.time()
        answer = await loop.run_in_executor(None, lambda: qa_chain.run(question))
        elapsed = time.time() - start
        return f"回答: {answer}\n\n响应时间: {elapsed:.2f} 秒"
    except Exception as e:
        return f"错误: {str(e)}"

iface = gradio.Interface(
    fn=ask_question,
    inputs=gradio.Textbox(label="输入问题", placeholder="在这里输入你的课程问题..."),
    outputs=gradio.Textbox(label="答案"),
    title="LearnMate DeepSeek RAG 测试",
    description="基于课程 PDF 的智能问答系统（MMR 检索，动态 k）",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()