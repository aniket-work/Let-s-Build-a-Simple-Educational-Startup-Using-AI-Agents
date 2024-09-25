from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from utils import setup_logging

logger = setup_logging()


def create_vector_store(documents, collection_name):
    try:
        logger.info(f"Creating vector store for {collection_name}")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name
        )
    except Exception as e:
        logger.error(f"Failed to create vector store for {collection_name}: {e}")
        raise