from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from os.path import exists


def create_vector_db(persist_directory, embedding_function):
    loader = PyPDFLoader("data/original_data/Genshin_Impact_full_story.pdf")
    pages = loader.load_and_split()

    spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    documents = spliter.split_documents(pages)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    return vectordb

def get_vector_db(embedding_function, persist_directory="data/vector_db"):
    if exists(persist_directory):
        vectordb = Chroma(
            embedding_function=embedding_function,
            persist_directory=persist_directory)
        
        return vectordb
    
    return create_vector_db(persist_directory, embedding_function)
