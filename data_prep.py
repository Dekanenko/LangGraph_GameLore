"""
Module for creating and loading a Chroma vector database from a PDF document.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

DEFAULT_PERSIST_DIR = "data/vector_db"
DEFAULT_PDF_PATH = "data/original_data/Genshin_Impact_full_story.pdf"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def _create_vector_db(
    embedding_function: Embeddings,
    pdf_path: str,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Chroma:
    """
    Creates a Chroma vector database from a PDF document.

    Args:
        embedding_function: The embedding function to use for documents.
        pdf_path: The path to the PDF file.
        persist_directory: The directory to save the vector database.
        chunk_size: The size of text chunks.
        chunk_overlap: The overlap between text chunks.

    Returns:
        The created Chroma vector database.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = text_splitter.split_documents(pages)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory,
    )

    return vectordb


def get_vector_db(
    embedding_function: Embeddings,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    pdf_path: str = DEFAULT_PDF_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Chroma:
    """
    Loads an existing Chroma vector database or creates a new one if it doesn't exist.

    Args:
        embedding_function: The embedding function to use.
        persist_directory: The directory of the vector database.
        pdf_path: The path to the PDF file (if creating a new DB).
        chunk_size: The chunk size for text splitting (if creating a new DB).
        chunk_overlap: The chunk overlap for text splitting (if creating a new DB).

    Returns:
        The Chroma vector database instance.
    """
    if os.path.exists(persist_directory):
        vectordb = Chroma(
            embedding_function=embedding_function, persist_directory=persist_directory
        )
        return vectordb

    return _create_vector_db(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
