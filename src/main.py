import os
import streamlit as st
from pdf_processing import load_pdf_documents, split_text
from embedding import generate_embeddings
from vector_store import create_vector_store
from rag import setup_rag_chain
from audio_generation import generate_audio
from utils import load_config, setup_logging
from langchain.schema import Document
import datetime

logger = setup_logging()
config = load_config()


def main():
    # Load and process PDFs
    tutorial_pages, history_pages = load_pdf_documents(config['pdf_directory'])

    # Split text into chunks
    tutorial_text_chunks = split_text(tutorial_pages)
    history_text_chunks = split_text(history_pages)

    # Add metadata
    tutorial_chunks_with_metadata = add_metadata(tutorial_text_chunks, "History Tutorial")
    history_chunks_with_metadata = add_metadata(history_text_chunks, "history Report")

    # Generate embeddings
    tutorial_texts = [chunk["text"] for chunk in tutorial_chunks_with_metadata]
    tutorial_embeddings = generate_embeddings(tutorial_texts)

    history_texts = [chunk["text"] for chunk in history_chunks_with_metadata]
    history_embeddings = generate_embeddings(history_texts)

    # Create vector stores
    tutorial_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in
                        tutorial_chunks_with_metadata]
    history_documents = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in
                       history_chunks_with_metadata]

    tutorial_vector_store = create_vector_store(tutorial_documents, "tutorial-local-rag")

    # Set up RAG chain
    rag_chain = setup_rag_chain(tutorial_vector_store)

    # Load questions from file
    with open("src/questions.txt", "r") as file:
        questions = [line.strip() for line in file.readlines()]

    # Streamlit UI
    st.set_page_config(page_title="Simple Educational App", page_icon=":chart_with_upwards_trend:")
    st.title("Simple Educational App")

    for i, question in enumerate(questions, 1):
        with st.expander(question):
            logger.info(f"Question: {question}")
            answer = rag_chain.invoke(question)
            st.write(answer)

            # Generate audio for the answer
            audio_file = generate_audio(answer, api_key="sk_d9ad200b3faa167dcaa8d734eb2f79842b2dfe59c9a691c6", filename=f"tutorial_answer_{i}")

            # Read the audio file and display it
            with open(audio_file, "rb") as audio:
                st.audio(audio.read(), format="audio/mp3")


def add_metadata(chunks, doc_title):
    metadata_chunks = []
    for chunk in chunks:
        metadata = {
            "title": doc_title,
            "author": "company",
            "date": str(datetime.date.today())
        }
        metadata_chunks.append({"text": chunk, "metadata": metadata})
    return metadata_chunks


def generate_audio(text, api_key, filename):
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)

    audio_stream = client.generate(
        text=text,
        model="eleven_turbo_v2",
        stream=True
    )

    audio_file = f"src/content/sample_data/audio/{filename}.mp3"
    with open(audio_file, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    logger.info(f"Audio file {audio_file} created")
    return audio_file


if __name__ == "__main__":
    main()
