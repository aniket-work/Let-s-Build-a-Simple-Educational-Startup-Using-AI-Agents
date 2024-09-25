import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import load_config, setup_logging

logger = setup_logging()
config = load_config()


def load_pdf_documents(pdf_directory):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    tutorial_pages, history_pages = [], []

    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        logger.info(f"Processing file: {file_path}")

        loader = PyPDFLoader(file_path=file_path)
        pages = loader.load()

        if pdf_file.startswith('PYRAMID'):
            tutorial_pages.extend(pages)
        elif pdf_file.startswith('ROME'):
            history_pages.extend(pages)

    return tutorial_pages, history_pages


def split_text(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    text_chunks = []
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        text_chunks.extend(chunks)
    return text_chunks
