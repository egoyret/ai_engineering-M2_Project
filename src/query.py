import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from build_index import COLLECTION_NAME, PERSIST_DIR, PROJECT_ROOT
import json
import warnings
from transformers import logging as tf_logging
from datetime import datetime

# Suppress non-critical warnings
tf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

load_dotenv() # Carga variable de entorno (api keys, etc)

OUTPUT_FILE = PROJECT_ROOT / "outputs" / "sample_queries"

# Inicializo el LLM que voy a usar para la RAG query
client_llm = OpenAI()  # Usa la api key de .env
model_llm = "gpt-5-mini"  # Modelo a usar para la RAG query


# Capturo el cliente de chroma que se creó en el directorio de persistencia
client = chromadb.PersistentClient(path=str(PERSIST_DIR))

# Usa la collection (vector store) que se creó en el build_index.py
try:
    collection_store = client.get_collection(
        name=COLLECTION_NAME)
    # print(f"[DEBUG] Se encontró la collection con {collection_store.count()} documentos")
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
    }}

"""

def rag_query(user_question):
    top_k = 3
    # Retrieval de los chunks relacionados con la pregunta
    results = collection_store.query(
        query_texts=[user_question],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]

    # print(f"[DEBUG] Question: {user_question} \n Retrieved {len(retrieved_docs)} chunks from the collection")

    # Construimos contexto pasandole los chunks recuperados
    context_block = ""
    for idx, doc in enumerate(retrieved_docs):
        context_block += f"\n[CHUNK {idx+1} | ID: {retrieved_ids[idx]}]\n{doc}\n"
        # print(f"[DEBUG]\n[CHUNK {idx+1} | ID: {retrieved_ids[idx]}]\n{doc}\n")

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
        model=model_llm,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=None
    )

    raw_output = response.choices[0].message.content

    # Validación de JSON
    try:
        parsed = json.loads(raw_output)
    except:
        parsed = {
            "user_question": user_question,
            "system_answer": "Error: salida no válida del modelo.",
            "chunks_related": retrieved_ids
        }

    return parsed

def agente_evaluador(query, answer, sources):
    prompt = f"""
    Eres un evaluador experto en sistemas RAG.

    Pregunta del usuario:
    {query}

    Respuesta generada:
    {answer}

    Contexto recuperado:
    {sources}

    Evalúa del 1 al 10:

    1. Relevancia (¿responde correctamente a la pregunta?)
    2. Fidelidad (¿está alineada con el contexto?)
    3. Claridad (¿es clara y bien redactada?)

    Devuelve SOLO un JSON válido con este formato:

    {{
      "relevancia": int,
      "fidelidad": int,
      "claridad": int,
      "justificación": "breve explicación"
    }}
    """

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

if __name__ == "__main__":

    # Aqui algunos queries de ejemplo:
    queries = [
        "¿Existe horario flexible?",
        "¿Cuántos días tengo si tengo 8 años en la empresa?",
        "¿de cuanto es el bono anual ?",
        "¿Puedo instalar software libre en mi laptop?",
        "¿Cuánto es el preaviso si renuncio?",
        "¿Me pagan horas extra?",
        "¿Cual es un buen framework para Python?",
        "¿Tengo seguro médico?"
    ]

    # Consulta manual:
    input_query = input("Ingrese la consulta (o 'ejemplos' para consultas de ejemplo' : ")
    if input_query and input_query.lower() != "ejemplos":
        result = rag_query(input_query)
        # Evaluacion de la respuesta
        evaluacion = agente_evaluador(result["user_question"],
                                      result["system_answer"],
                                      result["chunks_related"])
        result["evaluacion"] = evaluacion
        print("\n" + "***Resultado consulta manual ingresada***" + "\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n" + "=" * 70 + "\n")


    # Consultas de ejemplo (queda grabado el json de respuesta)
    generated_json = []
    if input_query.lower() == "ejemplos":
        for q in queries:
            result = rag_query(q)
            # Evaluacion de la respuesta
            evaluacion = agente_evaluador(result["user_question"],
                                          result["system_answer"],
                                          result["chunks_related"])
            result["evaluacion"] = evaluacion
            generated_json.append(result)

            # print(json.dumps(result, indent=2, ensure_ascii=False))
            # print("\n" + "="*70 + "\n")

    if generated_json:
        # ====== Nombre de archivo con timestamp ======
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_FILE}_{timestamp}.json"
        # Grabamos el JSON en un archivo
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(generated_json, f, ensure_ascii=False, indent=4)
        print(f"JSON generado y grabado en {filename}")

"""
      "relevance": int,
      "faithfulness": int,
      "clarity": int,
      "justification": "breve explicación"
"""