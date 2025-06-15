import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os
import tempfile

# Configuración de la página de Streamlit
st.set_page_config(page_title="Chatbot UOC RAG", layout="wide")
st.title("🤖 Chatbot UOC RAG")

# --- Funciones con Caché para la Lógica Principal ---

@st.cache_resource
def load_llm_and_embeddings():
    """Carga los embeddings y el LLM una sola vez."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm, embeddings

@st.cache_data
def process_pdf(_uploaded_file):
    """Procesa el PDF y devuelve los trozos (chunks)."""
    # Usamos un archivo temporal para que PyPDFLoader pueda leerlo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Limpiamos el archivo temporal después de cargarlo
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    docs_split = text_splitter.split_documents(documents)
    return docs_split

def create_retrieval_qa_chain(docs_split, llm, embeddings):
    """Crea la base de datos vectorial y la cadena de QA."""
    # Chroma DB se crea en memoria. Para apps complejas, se podría persistir.
    vectordb = Chroma.from_documents(docs_split, embedding=embeddings)
    # La creación del retriever y la cadena es rápida, no necesita caché pesada.
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain


# --- Lógica de la Aplicación ---

# Cargar modelos (esto solo se ejecutará la primera vez)
llm, embeddings = load_llm_and_embeddings()

# Subida de archivo
uploaded_file = st.file_uploader("📄 Sube un documento PDF de la UOC", type="pdf")

if uploaded_file is not None:
    # Procesar el PDF y crear la cadena de QA (usando caché)
    docs_split = process_pdf(uploaded_file)
    qa_chain = create_retrieval_qa_chain(docs_split, llm, embeddings)
    
    # Inicializar el historial del chat si no existe
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("💬 Pregunta sobre el documento:"):
        # Añadir pregunta del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar y mostrar la respuesta del chatbot
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = qa_chain.invoke({"query": prompt})
                st.markdown(response["result"])
        
        # Añadir respuesta del chatbot al historial
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

else:
    st.info("Por favor, sube un documento PDF para comenzar.")