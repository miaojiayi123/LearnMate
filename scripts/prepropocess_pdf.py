# LearnMate/scripts/preprocess_pdf.py
# 提取 PDF 中的文本内容
import fitz  # PyMuPDF
import os


# 配置输入输出路径
INPUT_DIR = "../data/"
OUTPUT_DIR = "../output/"
pdf_file = os.path.join(INPUT_DIR, "Chapter 2.pdf")

# 打开 PDF 并提取文本
doc = fitz.open(pdf_file)
text = ""
for page in doc:
    text += page.get_text()
print("提取完成，文本长度：", len(text))

# 保存提取结果
with open(os.path.join(OUTPUT_DIR, "extracted_text.txt"), "w", encoding="utf-8") as f:
    f.write(text)

import re

# 读取提取的文本
with open(os.path.join(OUTPUT_DIR, "extracted_text.txt"), "r", encoding="utf-8") as f:
    text = f.read()
# 清洗文本
cleaned_text = re.sub(r'\n+', ' ', text)  # 合并多行
cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', cleaned_text)  # 移除特殊字符，保留中文
print("清洗后文本长度：", len(cleaned_text))

# 保存清洗结果
with open(os.path.join(OUTPUT_DIR, "cleaned_text.txt"), "w", encoding="utf-8") as f:
    f.write(cleaned_text)


from langchain_text_splitters import RecursiveCharacterTextSplitter

# 读取清洗后的文本
with open(os.path.join(OUTPUT_DIR, "cleaned_text.txt"), "r", encoding="utf-8") as f:
    text = f.read()
# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    with open(os.path.join(OUTPUT_DIR, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
        f.write(chunk)
print(f"分块完成，总计 {len(chunks)} 个块")