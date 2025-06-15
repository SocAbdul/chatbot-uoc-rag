import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
import os

st.set_page_config(page_title="Chatbot UOC RAG", layout="wide")
st.title("📚 Chatbot UOC (RAG)")

uploaded_file = st.file_uploader("Sube un documento PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # 1. Cargar el PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # 2. Dividir el texto en partes
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = splitter.split_documents(documents)

    # 3. Embeddings (con HuggingFace, gratis)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Crear vectorstore
    vectordb = Chroma.from_documents(docs_split, embeddings)

    # 5. Configurar el modelo
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # Necesita clave OpenAI
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.success("✅ Documento procesado. Ya puedes hacer preguntas.")

    # 6. Chat
    user_question = st.text_input("Haz una pregunta sobre el documento:")
    if user_question:
        with st.spinner("Pensando..."):
            response = qa_chain.run(user_question)
            st.write("🧠 Respuesta:")
            st.markdown(response)

    # Limpieza del archivo temporal
    os.remove(tmp_path)
