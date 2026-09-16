"""
文档加载器 - 支持多种文档格式
使用 LlamaIndex 的 DocumentLoaders 加载文档
"""
import os
from pathlib import Path
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import PyMuPDFReader, DocxReader

from app.config.settings import settings


class DocumentLoader:
    """文档加载器 - 支持 PDF、DOCX、Markdown、TXT"""

    def __init__(self):
        self._readers = {
            ".pdf": PyMuPDFReader(),
            ".docx": DocxReader(),
        }

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文件
        Args:
            file_path: 文件路径
        Returns:
            Document 列表
        """
        ext = Path(file_path).suffix.lower()

        if ext in self._readers:
            documents = self._readers[ext].load_data(file=Path(file_path))
        elif ext in (".md", ".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents = [Document(text=content, metadata={"source": file_path})]
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 为每个文档添加源文件元数据
        for doc in documents:
            doc.metadata["source_file"] = file_path
            doc.metadata["filename"] = Path(file_path).name

        return documents

    def load_directory(self, dir_path: str) -> List[Document]:
        """
        加载目录下所有支持的文档
        Args:
            dir_path: 目录路径
        Returns:
            Document 列表
        """
        all_documents = []
        supported_ext = {".pdf", ".docx", ".md", ".txt"}

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if Path(file_path).suffix.lower() in supported_ext:
                try:
                    docs = self.load_file(file_path)
                    all_documents.extend(docs)
                    print(f"[DocumentLoader] 已加载: {file_name}")
                except Exception as e:
                    print(f"[DocumentLoader] 加载失败 {file_name}: {e}")

        print(f"[DocumentLoader] 共加载 {len(all_documents)} 个文档")
        return all_documents


class DocumentChunker:
    """文档分块器 - 将文档切分为适合检索的片段"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # 使用 LlamaIndex 的 SimpleNodeParser 进行分块
        self._parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_documents(self, documents: List[Document]) -> list:
        """
        将文档列表切分为节点（chunks）
        Args:
            documents: Document 列表
        Returns:
            切分后的节点列表
        """
        nodes = self._parser.get_nodes_from_documents(documents)
        print(f"[DocumentChunker] {len(documents)} 个文档 → {len(nodes)} 个片段")
        return nodes
