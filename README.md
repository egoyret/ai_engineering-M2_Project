# ai_engineering-M2_Project
AI Engineering Course proyecto modulo 2

## Consigna
Crea un chatbot de soporte para FAQs usando RAG que responda preguntas basándose en un documento del sistema. El sistema debe procesar un documento de texto plano, dividirlo en chunks (al menos 20 chunks) y generar embeddings. Para cada pregunta del usuario, devuelve una salida en JSON que contenga user_question, system_answer y chunks_related usados para generar la respuesta. Utiliza métodos de búsqueda vectorial (p. ej., k-NN, ANN, rango o híbridos) para encontrar de forma eficiente los chunks relevantes.


## Pasos

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/egoyret/ai_engineering-M2_Project.git
   ```

2. Creá y activá un entorno virtual (desde el root del proyecto):

   * **venv** (Linux/macOS/Windows PowerShell):

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate    # Unix/macOS
 
     ```
3. Instalá las dependencias:

   ```bash
   pip install -r requirements.txt
   ```
4. Agrega una API key en tu .env file (template de .env en .env.example en el root):
    ```bash
   OPENAI_API_KEY=<your api key>
      ```
   
## Estructura del proyecto

```
ai_engineering-M2_Project/
├── data/
│   ├── faq_document.txt  # Base de conocimeinto de RRHH
│
├── outputs/
│   ├── sample_queries_yyyy-mm-dd_hh-mm-ss.json # Outputs del chatbot    
├── src/
│   ├── build_index.py                           # Contruye y almacena el vector store
│   ├── query.py                                 # Ejecuta la consulta del usuario y genera la respuesta con RAG
│   ├── chroma_rrhh_db.py/                        # folder para el vector store que se crea al ajecutar build_index.py
```

## Ejecución:

1.Se sugiere abrir el faq_document.txt para ver el contenido del documento.
2.Leer el documento, generar los chunks y los embeddings y guardarlos en el vector store.
```
python3 src/build_index.py
```
3.Hacer una consulta al chatbot. Ante el prompt, tipear una consulta o el texto 'ejemplos' para preguntas precargadas
```
python3 src/query.py
```
Las consultas de ejemplo pueden modificarse editando la lista 'queries' en query.py. Las respuestas de estas consultas quedan guardadas en la carpeta data con nombre de archivo output_jason y la fecha de corrida.

## Tecnologias utilizadas:
Embeddings e indexado de vectores:

-Se utiliza Chroma para generar los vectores de embedding y guardarlos en un vector store, y para hacer las busquedas por similitud semántica.

-Para generar los embeddings en Chroma a traves de las 'embedding functions' usamos uno de los modelos de Hugging Face de sentence transformers (sentence-transformers/all-MiniLM-L6-v2).

-Con Chroma se genera un objecto 'collections' que tiene asociado el modelo de embeddings a utilizar y el directorio donde alacenar el vector store que genera.

-La 'collections' se usa tanto para los documentos de la base de conocimientos como para la consulta del usuario, de manera que se usa el mismo metodo de embedding.

Estrategia de chunking:

La base de conocimientos esta formada por una serie de preguntas y respuestas (FAQ). Cada una de ellas tiene como máximo unas 300 palabras (aprox 400 tokens) por lo que se decidió usar chunks de tamaño variable siendo cada chunk exactamente un par de respuesta y pregunta, lo que nos asegura de no perder información.

Modelo de LLM:

Para el RAG usamos un modelo de OpenAI al que le pasamos en el contexto los chunks previamente seleccionados por la 'collections' de Chroma.

Evaluación de consultas:

Se utiliza un LLM como agente evaluador para evaluar la calidad de las respuestas generadas por el chatbot.

