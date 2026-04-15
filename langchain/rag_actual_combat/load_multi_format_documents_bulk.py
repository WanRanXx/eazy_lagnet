from typing import List

from langchain_community.document_loaders import (
UnstructuredMarkdownLoader, TextLoader, PyPDFLoader
)
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_core.documents import Document
import os

def batch_load_documents(folder_path: str) -> List[Document]:
    """
    批量加载文件夹内的所有官方支持格式文档（基于新版加载器）
    :param folder_path: 知识库文件夹地址
    :return 所有文档的Document对象列表
    """

    all_docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            # 忽略文件夹
            continue
        try:
            if filename.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                print("不支持的文件格式，跳过文件：", filename)
                continue
            # 加载并添加文档
            doc = loader.load()
            all_docs.extend(doc)
            print(f"成功加载：{filename}，生成{len(doc)}个Document对象")
        except Exception as e:
            print(f"加载文件 {file_path} 时出错：{e}")
            continue
    return all_docs