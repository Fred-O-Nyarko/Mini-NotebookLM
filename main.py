import streamlit as st
import os
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from bs4 import BeautifulSoup
import requests
import tempfile

# --- Configuration & Setup ---
st.set_page_config(page_title="Mini-NotebookLM", layout="wide")

# 1. Setup Embeddings (Local)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embed_model = load_embeddings()

# 2. Semantic Chunking Logic
def process_text(text_content):
    text_splitter = SemanticChunker(embed_model)
    docs = text_splitter.create_documents([text_content])
    return docs

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)
    
    # Extract text and chunk semantically
    full_text = "\n".join([page.page_content for page in pages])
    return process_text(full_text)

def process_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = "\n".join([p.get_text() for p in paragraphs])
    return process_text(text_content)

# 3. Vector Store Management
@st.cache_resource
def get_vectorstore():
    # Persistent directory for ChromaDB
    persist_directory = "chroma_db"
    return Chroma(persist_directory=persist_directory, embedding_function=embed_model)

# 4. The RAG Engine (Gemini 3 Flash via google-genai)
def get_rag_response(query, vectorstore, api_key):
    # Strictly limit to top 3 chunks for precision
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever._get_relevant_documents(query, run_manager=CallbackManagerForRetrieverRun)
    
    if not context_docs:
        return "Nothing found in context"
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are a RAG assistant. Answer the question ONLY using the context below.
    If the answer is not in the context, say "Nothing found in context".
    
    Context: {context_text}
    Question: {query}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", # Assuming user meant latest available if Gemini 3 is not out or they meant 2.0
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
        )
    )
    return response.text

# --- Streamlit UI ---
st.title("Mini Notebook LM")

# API Key handling
api_key = st.sidebar.text_input("Enter Google API Key", type="password")

if api_key:
    vectorstore = get_vectorstore()
    

    # Sidebar for Ingestion
    with st.sidebar:
        st.header("Data Ingestion")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        url_input = st.text_input("Enter URL")
        
        if st.button("Index Data"):
            new_docs = []
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    new_docs.extend(process_pdf(uploaded_file))
            if url_input:
                with st.spinner("Scraping URL..."):
                    new_docs.extend(process_url(url_input))
            
            if new_docs:
                vectorstore.add_documents(new_docs)
                st.success(f"Successfully indexed {len(new_docs)} semantic chunks!")
            else:
                st.warning("No data provided to index.")

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about your indexed documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Analyzing context..."):
            try:
                response = get_rag_response(prompt, vectorstore, api_key)
            except Exception as e:
                response = f"Error calling Gemini API: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
else:
    st.info("Please enter your Google API Key in the sidebar to start.")
