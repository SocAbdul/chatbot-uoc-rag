import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("Chatbot RAG UOC")

if not OPENAI_API_KEY:
    st.error("❌ Por favor configura tu clave OPENAI_API_KEY en el archivo .env")
    st.stop()

uploaded_files = st.file_uploader(
    "Sube uno o varios PDFs con documentos UOC",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Procesando documentos..."):
        documents = []
        for pdf_file in uploaded_files:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_split = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        vectordb = Chroma.from_documents(docs_split, embeddings)

        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
            retriever=retriever
        )

        st.success("Documentos indexados correctamente. ¡Ya puedes hacer preguntas!")

        pregunta = st.text_input("Haz una pregunta sobre la UOC:")

        if pregunta:
            with st.spinner("Buscando respuesta..."):
                respuesta = qa_chain.run(pregunta)
                st.write(respuesta)
else:
    st.info("Sube uno o varios PDFs para comenzar.")
