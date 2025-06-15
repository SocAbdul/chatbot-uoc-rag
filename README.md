# 🤖 Chatbot UOC RAG

Este es un chatbot inteligente para estudiantes y profesores de la UOC. Permite hacer preguntas sobre la universidad utilizando información real contenida en documentos oficiales (PDFs, apuntes, exámenes, etc).

📚 Basado en tecnología RAG (Retrieval Augmented Generation) y LangChain.

## Tecnologías usadas
- Python 3.10
- Streamlit
- LangChain
- ChromaDB
- OpenAI (GPT-4 o GPT-3.5)
- dotenv

## Cómo ejecutar
```bash
conda activate chatbot-rag
pip install -r requirements.txt
streamlit run app.py

¿Cómo funciona?
El usuario escribe una pregunta.

El sistema busca la información relevante en los documentos.

El LLM genera una respuesta basada solo en datos internos (UOC).

⚠️ Nota
No usa Google ni datos externos para evitar errores o desinformación