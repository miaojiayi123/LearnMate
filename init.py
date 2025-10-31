# 文件路径：init.py
from core.learn_mate_core import preprocess_pdf, build_vectorstore
import os

if __name__ == "__main__":
    pdf_path = "data/Chapter 2.pdf"
    if not os.path.exists(pdf_path):
        print(f"错误：未找到 {pdf_path}")
    else:
        preprocess_pdf(pdf_path)
        build_vectorstore()
        print("LearnMate 初始化完成！")
        print("启动 API：uvicorn api.api:app --reload")