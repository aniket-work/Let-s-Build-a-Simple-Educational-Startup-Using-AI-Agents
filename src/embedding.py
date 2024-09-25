import ollama
from utils import load_config, setup_logging

logger = setup_logging()
config = load_config()


def generate_embeddings(text_chunks, model_name=config['embedding_model']):
    embeddings = []
    for chunk in text_chunks:
        embedding = ollama.embeddings(model=model_name, prompt=chunk)
        embeddings.append(embedding)
    return embeddings
