import os
import time
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from sentence_transformers import SentenceTransformer, CrossEncoder

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y CREDENCIALES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spyro Scientific RAG",
    page_icon="🐉",
    layout="centered"
)

load_dotenv()

SPYRO_AVATAR = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f409.png"
USER_AVATAR = "👤"

st.title("🐉 Spyro Scientific Assistant")
st.caption("Sistema Conversacional RAG | FAISS + Re-ranking (Cross-Encoder) + Gemini 2.5 Flash - Creado por Edison Quizhpe.")

# -----------------------------------------------------------------------------
# 2. CARGA EFICIENTE DE MODELOS Y ARTEFACTOS (Caché)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos pesados y base vectorial FAISS...")
def load_resources():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    faiss_index = faiss.read_index("archivo_faiss.index")
    return embedding_model, reranker, faiss_index

@st.cache_data(show_spinner="Cargando metadatos del corpus...")
def load_data():
    return pd.read_pickle("datos_corpus.pkl")

try:
    embedding_model, reranker, faiss_index = load_resources()
    df = load_data()
except Exception as e:
    st.error(f"❌ Error al cargar los artefactos locales: {e}")
    st.stop()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("api_key")
ai_client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------------------
# 3. PIPELINE RAG Y FUNCIONES AUXILIARES
# -----------------------------------------------------------------------------
def format_score(score_float):
    """Formatea dinámicamente el score para alta precisión."""
    pct = score_float * 100
    if 0 < pct < 0.01:
        return f"{pct:.4f}%"
    return f"{pct:.2f}%"

def retrieve_and_rerank(query, top_k_faiss=10, top_n_final=3):
    """Fase R: Búsqueda vectorial Coseno + Re-ranking con Cross-Encoder."""
    q_vec = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(q_vec)
    
    scores, indices = faiss_index.search(q_vec, top_k_faiss)
    candidates = [df.iloc[idx].to_dict() for idx in indices[0] if idx < len(df) and idx != -1]
    
    if not candidates:
        return []
    
    pairs = [[query, cand['text_to_embed']] for cand in candidates]
    raw_scores = reranker.predict(pairs)
    
    sigmoid_scores = 1 / (1 + np.exp(-raw_scores))
    for cand, score, raw in zip(candidates, sigmoid_scores, raw_scores):
        cand['rerank_score'] = float(score)
        cand['raw_score'] = float(raw)
        
    return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_n_final]


def generate_rag_response(query, contexts, max_retries=4):
    """Fase G: Generación Determinista con Citas de Documentos en Español."""
    if not contexts:
        return "El corpus no contiene información suficiente para responder a esta consulta."
    
    # Inyectar los IDs de los documentos para que Gemini pueda citarlos
    context_blocks = []
    for i, doc in enumerate(contexts):
        doc_tag = doc.get("doc_id", f"doc_{i+1}")
        context_blocks.append(f"Documento [{doc_tag}]:\nTítulo: {doc.get('title', '')}\nTexto: {doc['text_to_embed']}")
        
    context_text = "\n\n".join(context_blocks)
    
    system_instruction = (
        "Eres un asistente científico experto. Lee y analiza el contexto proporcionado (en inglés) "
        "y responde a la pregunta del usuario estrictamente en ESPAÑOL utilizando ÚNICAMENTE el contexto.\n\n"
        "Reglas obligatorias:\n"
        "1. La respuesta final DEBE estar redactada completamente en ESPAÑOL.\n"
        "2. Cita la fuente al final de cada afirmación relevante usando el identificador entre corchetes, por ejemplo: [doc_1] o [doc_1554].\n"
        "3. Si el contexto NO contiene información suficiente para responder la consulta, DEBES responder exactamente: "
        "'El corpus no contiene información suficiente para responder a esta consulta.'\n"
        "4. No inventes ni utilices conocimiento externo al contexto."
    )
    
    prompt = f"Context:\n{context_text}\n\nQuestion / Consulta: {query}\n\nAnswer (en español con citas [doc_id]):"
    
    for attempt in range(max_retries):
        try:
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0
                )
            )
            return response.text
        except APIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 12 * (attempt + 1)
                st.toast(f"⏳ Límite de API alcanzado. Reintentando en {wait_time}s ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                return f"⚠️ Error al conectar con la API de Gemini: {e}"
                
    return "Error: Se superó el límite de reintentos por saturación en la API."


def render_evidences_table(docs):
    """Función para construir y mostrar la tabla interactiva de evidencias al estilo DataFrame."""
    table_rows = []
    for idx, doc in enumerate(docs):
        doc_identifier = doc.get("doc_id", f"doc_{idx+1}")
        categories = doc.get("categories", doc.get("categoria", "cs.AI / cs.LG"))
        
        table_rows.append({
            "doc_id": doc_identifier,
            "score_rerank": format_score(doc.get("rerank_score", 0)),
            "titulo": doc.get("title", "Sin título"),
            "categorias": str(categories)
        })
    
    # Crear DataFrame de Pandas para Streamlit
    df_evidences = pd.DataFrame(table_rows)
    st.dataframe(
        df_evidences,
        use_container_width=True,
        hide_index=True
    )

# -----------------------------------------------------------------------------
# 4. MEMORIA Y COMPONENTES VISUALES DEL CHAT
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "¡Hola! Soy **Spyro**, tu asistente científico. Hazme cualquier consulta sobre el corpus y te responderé en español con citas directas a las evidencias.",
            "sources": []
        }
    ]

# Renderizar historial de mensajes
for message in st.session_state.messages:
    avatar = SPYRO_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        if message.get("sources"):
            with st.expander(" Ver evidencias recuperadas"):
                render_evidences_table(message["sources"])

# -----------------------------------------------------------------------------
# 5. ENTRADA DE USUARIO Y CICLO DE CONVERSACIÓN
# -----------------------------------------------------------------------------
if user_input := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=SPYRO_AVATAR):
        with st.spinner("Spyro está analizando los artículos científicos..."):
            retrieved_docs = retrieve_and_rerank(user_input, top_n_final=3)
            answer = generate_rag_response(user_input, retrieved_docs)
            
            st.markdown(answer)
            
            if retrieved_docs and "El corpus no contiene" not in answer:
                with st.expander(" Ver evidencias recuperadas"):
                    render_evidences_table(retrieved_docs)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": retrieved_docs if "El corpus no contiene" not in answer else []
    })