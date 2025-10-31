# LearnMate - 个性化课程学习伙伴

基于 **课程章节文档** 的 **RAG 智能问答系统**。

## 功能
- PDF 自动预处理（PyMuPDF + LangChain）
- 向量检索（Chroma + MMR）
- 大模型推理（DeepSeek）
- 持久化缓存（diskcache，0.01 秒响应）
- REST API（FastAPI + OpenAPI）

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/miaojiayi123/LearnMate.git
cd LearnMate

# 2. 安装
pip install -r requirements.txt

# 3. 放入 PDF
mkdir input && cp "你的PDF路径/Chapter 2.pdf" input/

# 4. 初始化
python init.py

# 5. 启动 API
uvicorn api.api:app --reload

访问：http://127.0.0.1:8000/docs