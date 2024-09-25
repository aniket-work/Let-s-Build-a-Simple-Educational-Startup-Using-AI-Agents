# general
import os
import datetime
import ollama
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the directory where your PDFs are stored
pdf_directory = "content"
save_dir = pdf_directory

from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader

# Get a list of all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Initialize lists to hold pages from Nvidia and Tesla PDFs separately
nvidia_pages = []
tesla_pages = []

# Iterate through each PDF file and load it
for pdf_file in pdf_files:
    file_path = os.path.join(pdf_directory, pdf_file)
    logger.info(f"Processing file: {file_path}")

    # Load the PDF and split it into pages
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()

    # Categorize pages based on the PDF filename
    if pdf_file.startswith('NVIDIA'):
        nvidia_pages.extend(pages)
    elif pdf_file.startswith('invoice_2'):
        tesla_pages.extend(pages)

# print out the first page of the first document for each category as an example
if nvidia_pages:
    logger.info("First page of the first Nvidia document:")
    logger.info(nvidia_pages[0].page_content[:500])  # Print first 500 characters
else:
    logger.info("No Nvidia pages found in the PDFs.")

if tesla_pages:
    logger.info("First page of the first Tesla document:")
    logger.info(tesla_pages[0].page_content[:500])  # Print first 500 characters
else:
    logger.info("No Tesla pages found in the PDFs.")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

# Split text into chunks for Nvidia pages
nvidia_text_chunks = []
for page in nvidia_pages:
    chunks = text_splitter.split_text(page.page_content)
    nvidia_text_chunks.extend(chunks)

# Split text into chunks for Tesla pages
tesla_text_chunks = []
for page in tesla_pages:
    chunks = text_splitter.split_text(page.page_content)
    tesla_text_chunks.extend(chunks)


# Example metadata management (customize as needed)
def add_metadata(chunks, doc_title):
    metadata_chunks = []
    for chunk in chunks:
        metadata = {
            "title": doc_title,
            "author": "company",  # Update based on document data
            "date": str(datetime.date.today())
        }
        metadata_chunks.append({"text": chunk, "metadata": metadata})
    return metadata_chunks


# Add metadata to Nvidia chunks
nvidia_chunks_with_metadata = add_metadata(nvidia_text_chunks, "NVIDIA Financial Report")

# Add metadata to Tesla chunks
tesla_chunks_with_metadata = add_metadata(tesla_text_chunks, "TESLA Financial Report")


# Function to generate embeddings for text chunks
def generate_embeddings(text_chunks, model_name='nomic-embed-text'):
    embeddings = []
    for chunk in text_chunks:
        # Generate the embedding for each chunk
        embedding = ollama.embeddings(model=model_name, prompt=chunk)
        embeddings.append(embedding)
    return embeddings


# Example: Embed Nvidia text chunks
nvidia_texts = [chunk["text"] for chunk in nvidia_chunks_with_metadata]
nvidia_embeddings = generate_embeddings(nvidia_texts)

logger.info(f"Generated {len(nvidia_embeddings)} embeddings for Nvidia text chunks")

# Example: Embed Tesla text chunks
tesla_texts = [chunk["text"] for chunk in tesla_chunks_with_metadata]
tesla_embeddings = generate_embeddings(tesla_texts)

logger.info(f"Generated {len(tesla_embeddings)} embeddings for Tesla text chunks")

# Wrap Nvidia texts with their respective metadata into Document objects
nvidia_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in
                    nvidia_chunks_with_metadata]


# Function to create vector store
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


# Wrap Tesla texts with their respective metadata into Document objects
tesla_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in
                   tesla_chunks_with_metadata]

# Create vector stores
try:
    nvidia_vector_store = create_vector_store(nvidia_documents, "nvidia-local-rag")
    logger.info("Successfully created Nvidia vector store")
except Exception as e:
    logger.error(f"Failed to create Nvidia vector store: {e}")

try:
    tesla_vector_store = create_vector_store(tesla_documents, "tesla-local-rag")
    logger.info("Successfully created Tesla vector store")
except Exception as e:
    logger.error(f"Failed to create Tesla vector store: {e}")

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# LLM from Ollama
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    retriever=nvidia_vector_store.as_retriever(),
    llm=llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Example usage
questions = [
    "What are the main revenue drivers for Nvidia this fiscal year?",
    "Can you provide some financial advice on Nvidia Stock for the future? Should people consider buying it?",
    "What has Nvidia done on June 7, 2024?",
    "You can read: On June 7, 2024, we completed a 10-for-1 forward stock split. All share and per share amounts presented have been retroactively adjusted to reflect the stock split. Don't you?",
    "Is Nvidia financially a strong company and do you think its stock will rise?"
]

for question in questions:
    logger.info(f"Question: {question}")
    answer = chain.invoke(question)
    logger.info(f"Answer: {answer}\n")


# Audio generation section (commented out as it requires an API key)
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key="sk_d9ad200b3faa167dcaa8d734eb2f79842b2dfe59c9a691c6")

questions_for_audio = [
    "Can you provide some financial advice on Nvidia Stock for the future? Should people consider buying it?",
    "Vind je Nvidia financieel een sterk bedrijf en denk je dat zijn aandeel gaat stijgen?",
]

for i, question in enumerate(questions_for_audio, 1):
    audio_stream = client.generate(
        text=question,
        model="eleven_turbo_v2" if i <= 2 else "eleven_multilingual_v2",
        stream=True
    )

    with open(f"src/content/sample_data/audio/nvidia{i}.mp3", "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    logger.info(f"Audio file nvidia{i}.mp3 created")

logger.info("Script execution completed.")