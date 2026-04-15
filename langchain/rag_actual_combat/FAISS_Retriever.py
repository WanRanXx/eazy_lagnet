import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from langchain.rag_actual_combat.embedding_test import vector_db

embedding_model_name = r"D:\data\langchain\eazy_Iagent\tools\models\Qwen\Qwen3-Embedding-0___6B"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

try:
    vector_db = FAISS.load_local(
        folder_path="./faiss_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        index_name="local_cpu_faiss_index"
    )
    print("向量数据库加载成功")
except FileNotFoundError:
    raise FileNotFoundError("未找到./faiss_db本地文件")
except Exception as e:
    raise RuntimeError(f"加载FAISS向量库失败：{str(e)}")


# 创建检索器（v0.1+ 规范）
retriever: BaseRetriever = vector_db.as_retriever(
    search_kwargs={"k": 3},
)

query = "vllm特点"


try:
    retriever_docs: list[Document] = retriever.invoke(query)
    print(f"\n检索到相关片段{len(retriever_docs)}个：")
    for i, doc in enumerate(retriever_docs):
        print(f"\n片段{i + 1}：")
        print(f"内容：{doc.page_content}")
        print(f"来源文件：{doc.metadata.get('source', '未知')}")

    docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\n片段{i + 1}（相关性评分：{round(score, 4)}）：")
        print(f"内容：{doc.page_content}")
except Exception as e:
    raise RuntimeError(f"检索向量库失败：{str(e)}")