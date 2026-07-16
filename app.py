import os
import time
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError
from sentence_transformers import SentenceTransformer, CrossEncoder
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y CREDENCIALES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spyro Scientific RAG - Creado por Edison Quizhpe",
    page_icon="🐉",
    layout="centered"
)

load_dotenv()

SPYRO_AVATAR = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f409.png"
USER_AVATAR = "👤"

st.title("🐉 Spyro Scientific Assistant")
st.caption("Sistema Conversacional RAG | FAISS + Re-ranking (Cross-Encoder) + Gemini 2.0 Flash")

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

# Lectura de la API Key (Soporta entorno local y Streamlit Cloud)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("🔑 No se encontró la API Key de Gemini (`GEMINI_API_KEY`).")
    st.stop()

ai_client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------------------
# 3. PIPELINE RAG (Traducción + Recuperación + Re-ranking + Gemini)
# -----------------------------------------------------------------------------
def traducir_consulta_a_ingles(query):
    """Traduce la consulta a inglés únicamente para que FAISS y el Re-ranker la reconozcan."""
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Translate the following search query to English. Return ONLY the translated string without extra quotes or formatting: {query}"
        )
        return response.text.strip()
    except Exception:
        return query


def retrieve_and_rerank(query, fetch_k=20, top_n_final=3):
    """Fase R: Recupera amplia en FAISS y filtra con Cross-Encoder."""
    q_vec = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(q_vec)
    
    scores, indices = faiss_index.search(q_vec, fetch_k)
    candidatos = [df.iloc[idx].to_dict() for idx in indices[0] if idx < len(df) and idx != -1]
    
    if not candidatos:
        return []
    
    pairs = [[query, cand['text_to_embed']] for cand in candidatos]
    raw_scores = reranker.predict(pairs)
    
    # Sigmoide para calcular porcentaje de relevancia
    sigmoid_scores = 1 / (1 + np.exp(-raw_scores))
    for cand, score in zip(candidatos, sigmoid_scores):
        cand['rerank_score'] = float(score)
        
    return sorted(candidatos, key=lambda x: x['rerank_score'], reverse=True)[:top_n_final]


@retry(
    retry=retry_if_exception_type((ServerError, APIError)), 
    wait=wait_exponential(multiplier=2, min=2, max=10), 
    stop=stop_after_attempt(3)
)
def ejecutar_llamada_gemini(prompt, system_instruction):
    response = ai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0
        )
    )
    return response.text


def generate_rag_response(query, contexts):
    """Fase G: Generación fluida en español sin marcas invasivas."""
    if not contexts:
        return "El corpus no contiene información suficiente para responder a esta consulta."
    
    context_text = "\n\n".join([f"Documento {i+1}:\n{doc['text_to_embed']}" for i, doc in enumerate(contexts)])
    
    system_instruction = (
        "Eres un asistente científico experto. Lee y analiza el contexto proporcionado (en inglés) "
        "y responde a la pregunta del usuario estrictamente en ESPAÑOL utilizando ÚNICAMENTE el contexto.\n\n"
        "Reglas estrictamente obligatorias:\n"
        "1. La respuesta final DEBE estar redactada completamente en ESPAÑOL de manera clara, fluida y profesional.\n"
        "2. Basa tu respuesta de manera precisa en la información de los documentos.\n"
        "3. Si el contexto NO contiene información suficiente para responder la consulta de manera certera, DEBES responder exactamente: "
        "'El corpus no contiene información suficiente para responder a esta consulta.'\n"
        "4. No inventes, no asumas ni utilices conocimiento externo al contexto."
    )
    
    prompt = f"Context:\n{context_text}\n\nConsulta / Pregunta: {query}\n\nRespuesta (en español):"
    
    try:
        return ejecutar_llamada_gemini(prompt, system_instruction)
    except Exception:
        return "El corpus no contiene información suficiente para responder a esta consulta."

# -----------------------------------------------------------------------------
# 4. MEMORIA Y COMPONENTES VISUALES DEL CHAT
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "¡Hola! Soy **Spyro**, tu asistente científico. Hazme cualquier consulta sobre el corpus y te responderé en español respaldado por evidencias exactas.",
            "sources": []
        }
    ]

# Renderizar historial
for message in st.session_state.messages:
    avatar = SPYRO_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        if message.get("sources"):
            with st.expander("📚 Ver Evidencias Científicas Consultadas"):
                for idx, doc in enumerate(message["sources"]):
                    st.write(f"**[{idx+1}] {doc.get('title', 'Sin título')}** (Relevancia: {doc['rerank_score']*100:.2f}%)")
                    st.caption(f"{doc.get('abstract', doc.get('text_to_embed', ''))[:250]}...")

# -----------------------------------------------------------------------------
# 5. ENTRADA DE USUARIO Y CICLO DE CONVERSACIÓN
# -----------------------------------------------------------------------------
if user_input := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=SPYRO_AVATAR):
        with st.spinner("Spyro está analizando los artículos científicos..."):
            # 1. Traducir consulta al inglés únicamente para la búsqueda en FAISS
            search_query = traducir_consulta_a_ingles(user_input)
            
            # 2. Recuperar y reordenar usando la consulta traducida
            retrieved_docs = retrieve_and_rerank(search_query, fetch_k=20, top_n_final=3)
            
            # 3. Generar la respuesta formal con Gemini
            answer = generate_rag_response(user_input, retrieved_docs)
            
            st.markdown(answer)
            
            if retrieved_docs and "El corpus no contiene" not in answer:
                with st.expander("📚 Ver Evidencias Científicas Consultadas"):
                    for idx, doc in enumerate(retrieved_docs):
                        st.write(f"**[{idx+1}] {doc.get('title', 'Sin título')}** (Relevancia: {doc['rerank_score']*100:.2f}%)")
                        st.caption(f"{doc.get('abstract', doc.get('text_to_embed', ''))[:250]}...")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": retrieved_docs if "El corpus no contiene" not in answer else []
    })