import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

text_path = os.path.join("knowledge_base", "test.txt")
if not os.path.exists(text_path):
    raise FileNotFoundError("文件不存在，请检查文件路径是否正确")

# 加载文本文件
loader = TextLoader(text_path, encoding="utf-8")
txt_docs: list[Document] = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", " ", ""],
)

split_docs: list[Document] = text_splitter.split_documents(txt_docs)
# 加载本地向量模型
embedding_model_name = r"D:\data\langchain\eazy_Iagent\tools\models\Qwen\Qwen3-Embedding-0___6B"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
# 4. 构建并持久化FAISS向量库
try:
    # 生成向量并初始化FAISS（本地CPU计算，首次运行会下载模型，需联网）
    vector_db = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings,
    )

    # 持久化向量库到本地
    vector_db.save_local(
        folder_path="./faiss_db",
        index_name="local_cpu_faiss_index"  # 索引名改为本地CPU版
    )
    print("向量存储完成！向量数据已保存到 ./faiss_db 文件夹")
except Exception as e:
    raise RuntimeError(f"构建/保存向量库失败：{str(e)}")

# 5. 相似性检索测试
query = "vLLM是什么？"
try:
    # 一次性获取带评分的检索结果
    retrieved_docs_with_scores = vector_db.similarity_search_with_score(query, k=3)

    print(f"\n与问题「{query}」最相关的3个文本片段：")
    for i, (doc, score) in enumerate(retrieved_docs_with_scores):
        print(f"\n片段{i + 1}：")
        print(f"内容：{doc.page_content}")
        print(f"相关性评分（越小越相似）：{round(score, 4)}")
        print(f"来源：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"检索向量库失败：{str(e)}")