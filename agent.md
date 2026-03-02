1. Agent Ruleset: The "Verification First" Protocol
As an AI architect, I define the agent's behavior through a strict system prompt. This ensures the model remains a grounded assistant rather than a creative writer.

Agent Name: Verifier-RAG Agent
Role: Senior Technical Analyst & QA Assistant
Core Mandate: Act as a closed-loop retrieval system. Your primary goal is to synthesize answers strictly from the provided context.

Operational Constraints:

Scope Lockdown: You are strictly forbidden from using internal pre-trained knowledge to answer questions. If the answer is not in the provided snippets, respond with: Nothing found in context.

No Extrapolation: Do not offer "likely" scenarios or "general advice" unless the context explicitly contains it.

Source Integrity: If the context is ambiguous, state the ambiguity rather than guessing.

Tone: Professional, concise, and technical.

2. Technical Implementation Plan
Phase 1: Environment & Dependencies
We’ll use a modular structure. Ensure you have your Google API Key for Gemini.

Bash
pip install streamlit google-genai pypdf beautifulsoup4 langchain langchain-community \
            langchain-google-genai chromadb sentence-transformers
Phase 2: Data Ingestion (The Pipeline)
We use pypdf for PDFs and BeautifulSoup4 for web scraping. To meet the "Semantic Chunking" requirement, we use LangChain’s experimental semantic chunker which splits text based on meaning rather than character count.

PDF Logic: Iterate through pages using PyPDFLoader.

URL Logic: Use requests + BeautifulSoup to pull clean text from <p> tags.

Semantic Chunking: This uses the all-MiniLM-L6-v2 embeddings to determine where a topic shifts.

Phase 3: Vector Storage & Retrieval
Embeddings: HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2").

Vector Store: Chroma initialized with a persist_directory.

LLM Integration: ChatGoogleGenerativeAI(model="gemini-1.5-flash").

Phase 4: Streamlit UI & Context Management
Streamlit will maintain the chat history in st.session_state.messages to preserve the conversation flow.

3. Core Implementation (Python)
Python
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
import requests

# 1. Setup Embeddings (Local)
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Semantic Chunking Logic
def process_data(text_content):
    # Splits based on semantic similarity of sentences
    text_splitter = SemanticChunker(embed_model)
    docs = text_splitter.create_documents([text_content])
    return docs

# 3. The RAG Engine
def get_rag_response(query, vectorstore):
    # Strictly limit to top 3 chunks for precision
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.get_relevant_documents(query)
    
    if not context_docs:
        return "Nothing found in context"
    
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Strict Prompting to enforce the constraint
    prompt = f"""
    You are a RAG assistant. Answer the question ONLY using the context below.
    If the answer is not in the context, say "Nothing found in context".
    
    Context: {context_text}
    Question: {query}
    """
    
    response = llm.invoke(prompt)
    return response.content

# --- Streamlit UI ---
st.title("AI Notebook: Fintech-Grade RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Ingestion
with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    url_input = st.text_input("Enter URL")
    
    if st.button("Index Data"):
        # Logic for PDF or URL parsing goes here...
        st.success("Data Semantically Chunked & Stored in ChromaDB!")

# Chat Interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Placeholder: In a real app, 'vectorstore' would be loaded from local ChromaDB
    # response = get_rag_response(prompt, vectorstore)
    response = "This is where the RAG engine would process the query against ChromaDB."
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)