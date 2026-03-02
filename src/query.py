import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from build_index import COLLECTION_NAME, PERSIST_DIR
import json
import warnings
from transformers import logging as tf_logging

# Suppress non-critical warnings
tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

load_dotenv()


# Inicializo el LLM que voy a usar para la RAG query
client_llm = OpenAI()  # Usa el modelo default de .env


# Capturo el cliente de chroma que se creo en el directorio de persistencia
client = chromadb.PersistentClient(path=str(PERSIST_DIR))

# Usa la collection (vector store) que se creó en el build_index.py
try:
    collection_store = client.get_collection(
        name=COLLECTION_NAME)
    print(f"[DEBUG] Se encontró la collection con {collection_store.count()} documentos")
except Exception as e:
    print(f"Error: No pudo encontrar la collection '{COLLECTION_NAME}'. Asegurarse de que build_index.py se haya ejecutado antes.")
    print(f"Detalles: {e}")
    raise

# Definimos el system prompt para el LLM
SYSTEM_PROMPT = """
Eres un asistente de Recursos Humanos.

Reglas obligatorias:
1. Responde únicamente utilizando la información del contexto proporcionado.
2. Si la respuesta no está en el contexto, indica que no está disponible.
3. No inventes información.
4. Responde en formato JSON estrictamente:
    {{
      "user_question": "{user_question}",
      "system_answer": "tu respuesta aquí",
      "chunks_related": ["id y texto de los fragmentos usados. Ejemplo: HR-002: horario flexible"]
      "semantics_scores": [0.1, 0.2, 0.3, 0.4, 0.5],
      "confidence_score": 0.8
    }}

"""

def rag_query(user_question):
    top_k = 3
    # Retrieval
    results = collection_store.query(
        query_texts=[user_question],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]
    distances = results["distances"][0]

    # Convertimos distancia en similitud
    similarities = [round(1 - d, 4) for d in distances]

    # Construimos contexto
    context_block = ""
    for idx, doc in enumerate(retrieved_docs):
        context_block += f"\n[CHUNK {idx+1} | ID: {retrieved_ids[idx]}]\n{doc}\n"

    # Prompt final
    user_prompt = f"""
CONTEXTO:
{context_block}

PREGUNTA DEL USUARIO:
{user_question}

Recuerda devolver SOLO JSON válido.
"""

    # LLM call
    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content

    # Validación de JSON
    try:
        parsed = json.loads(raw_output)
    except:
        parsed = {
            "user_question": user_question,
            "system_answer": "Error: salida no válida del modelo.",
            "chunks_related": retrieved_ids,
            "semantic_scores": similarities,
            "confidence_score": 0.0
        }

    return parsed

if __name__ == "__main__":

    queries = [
        "¿Existe horario flexible?",
        "¿Cuántos días tengo si tengo 8 años en la empresa?",
        # "¿Puedo instalar software libre en mi laptop?",
        # "¿Cuánto es el preaviso si renuncio?",
        # "¿Me pagan horas extra?"
    ]

    for q in queries:
        result = rag_query(q)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n" + "="*70 + "\n")