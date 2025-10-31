import os
import time

import fitz
import gradio
import asyncio
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re

from functools import lru_cache
import diskcache as dc
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ------------------- 路径配置 -------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(base_dir, "input")
OUTPUT_DIR = os.path.join(base_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
persist_directory = os.path.join(OUTPUT_DIR, "chroma_db")
CACHE_DIR = os.path.join(OUTPUT_DIR, "rag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------- 缓存初始化 -------------------
rag_cache = dc.Cache(CACHE_DIR)


# ------------------- PDF 预处理函数 -------------------
def preprocess_pdf(pdf_path: str):
    """从 PDF 提取 → 清洗 → 分块 → 存入 output/"""
    print(f"正在处理 PDF: {pdf_path}")

    # 1. 提取文本
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    with open(os.path.join(OUTPUT_DIR, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"提取完成，长度: {len(text)}")

    # 2. 清洗
    cleaned_text = re.sub(r'\n+', ' ', text)
    cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', cleaned_text)
    with open(os.path.join(OUTPUT_DIR, "cleaned_text.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"清洗完成，长度: {len(cleaned_text)}")

    # 3. 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(cleaned_text)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(OUTPUT_DIR, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)
    print(f"分块完成，总计 {len(chunks)} 个块")
    return chunks


# ------------------- 构建向量库 -------------------
def build_vectorstore():
    chunks = []
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("chunk_") and f.endswith(".txt"):
            with open(os.path.join(OUTPUT_DIR, f), "r", encoding="utf-8") as file:
                chunks.append(file.read())

    if not chunks:
        raise FileNotFoundError("未找到分块文件")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory=persist_directory)
    print("Chroma 数据库构建完成")
    return vectorstore


# ------------------- 初始化 RAG 组件 -------------------
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

llm = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.1,
    max_tokens=500
)

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

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 15, "fetch_k": 30}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)


# ------------------- 缓存 RAG 函数 -------------------
def cached_rag(query: str) -> str:
    normalized = query.strip().lower()
    cache_key = f"rag_v1:{hashlib.md5(normalized.encode()).hexdigest()}"

    if cache_key in rag_cache:
        print(f"[缓存命中] {cache_key[-8:]}")
        return rag_cache[cache_key]

    print(f"[缓存未命中] 执行 RAG...")
    result = qa_chain.invoke({"query": query})["result"]
    rag_cache[cache_key] = result
    return result


# ------------------- 健康检查 -------------------
def health_check():
    return {
        "status": "healthy",
        "cache_size": len(rag_cache),
        "vector_count": vectorstore._collection.count()
    }