from pathlib import Path
import re # regular expressions
from dotenv import load_dotenv
import warnings
from transformers import logging as tf_logging
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

# Following is required to avoid display of harmless warning message from Chroma
tf_logging.set_verbosity_error()  # Suppress non-critical warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "faq_document.txt"
PERSIST_DIR = PROJECT_ROOT / "src" / "chroma_rrhh_db"
COLLECTION_NAME = "informacion_rrhh"
MODEL_NAME = "all-MiniLM-L6-v2"





def inicializar_chroma():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME)

    # Instancio Chroma y le defino el folder donde debe guardar el vector store
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    # Creo el dataset para los embeddings que se van a generar con el modelo definido.
    try: # Primero borro el dataset si existe
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function)

    return collection

# Leer el archivo de conocimiento y chunkearlo
def load_and_chunk(file_path):
    """
    Se crean chunks de largo variable aprovechando el separador que existe entre cada FAQ

    Formato de la data en la base de conocimiento:

    DOCUMENT_ID: HR-057
    CATEGORY: General
    Pregunta: ¿Quién actualiza las políticas?
    Respuesta: El área de Recursos Humanos.

---
    """
    text = Path(file_path).read_text(encoding="utf-8")
    raw_chunks = [c.strip() for c in text.split('---') if c.strip()]
    return raw_chunks

# Extraemos metadata usando regular expressions
def parse_chunk(chunk):
    doc_id = re.search(r'DOCUMENT_ID:\s*(.+)', chunk).group(1).strip()
    category = re.search(r'CATEGORY:\s*(.+)', chunk).group(1).strip()

    return {
        "id": doc_id,
        "category": category,
        "content": chunk
    }

# Generar embeddings, crear y almacenar un vector store
def genera_y_almacena_embeddings(chunks, collection):
    ids = []
    metadatas = []
    documents = []

    for chunk in chunks:
        metadata = parse_chunk(chunk)
        ids.append(metadata["id"])
        metadatas.append({"category": metadata["category"]})
        documents.append(metadata["content"])

    # Add all documents in batch for better persistence
    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents
    )
    # print(f"[DEBUG] Embeddings: {collection.count()} - Chunks: {len(chunks)}")

def main():

    # Eliminar directorio de base de datos antiguo para evitar problemas de datos antiguos
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        # print(f"[DEBUG] Directorio de base de datos antiguo eliminado en {PERSIST_DIR}")

    chunks = load_and_chunk(DATA_FILE)
    collection = inicializar_chroma()
    genera_y_almacena_embeddings(chunks, collection)

    print(f"Index created successfully! with {collection.count()} documents.")



if __name__ == "__main__":
    main()