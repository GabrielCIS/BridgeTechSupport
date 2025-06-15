# === 📚 AI Chatbot for Nonprofit Docs + Web ===
# This chatbot uses documents from Google Drive (Docs, PDFs, Sheets), 
# loads them into a vector store using LangChain, and uses OpenAI for Q&A.

import os
import json
import streamlit as st
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from googleapiclient.discovery import build
from google.oauth2 import service_account
import requests
import fitz  # PyMuPDF for PDFs
from langchain.schema import Document

# === CONFIGURATION ===
FOLDER_ID = "1z_zzdbB4zJo70o3rofTqwm30ux9dpRsX"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_KEY

# Parse Google credentials from secrets
google_creds = json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    google_creds,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

# === UTIL: Load text from files ===
def load_text_from_file(filepath):
    if filepath.endswith(".pdf"):
        doc = fitz.open(filepath)
        return "\n".join([page.get_text() for page in doc])
    elif filepath.endswith(".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            return "\n".join(line.strip() for line in f if line.strip())
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

# === Download files from Google Drive ===
@st.cache_resource
def download_google_docs(folder_id, _credentials):
    service = build('drive', 'v3', credentials=_credentials)
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        pageSize=100,
        fields="files(id, name, mimeType)"
    ).execute()

    files = results.get('files', [])
    paths = []

    for file in files:
        file_id = file['id']
        name = file['name']
        mime_type = file['mimeType']

        # Handle Google Docs
        if mime_type == 'application/vnd.google-apps.document':
            request = service.files().export_media(fileId=file_id, mimeType='text/plain')
            path = f"/tmp/{name}.txt"

        # Handle Google Sheets
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            request = service.files().export_media(fileId=file_id, mimeType='text/csv')
            path = f"/tmp/{name}.csv"

        # Handle PDFs
        elif mime_type == 'application/pdf':
            request = service.files().get_media(fileId=file_id)
            path = f"/tmp/{name}.pdf"

        else:
            continue  # Unsupported file type

        with open(path, 'wb') as f:
            f.write(request.execute())
        paths.append(path)

    return paths

# === Build vector store from documents ===
@st.cache_resource
def build_vector_store(docs_paths):
    documents = []
    for path in docs_paths:
        text = load_text_from_file(path)
        documents.append(Document(page_content=text, metadata={"source": path}))
    return Chroma.from_documents(documents, OpenAIEmbeddings())

# === Web search fallback using SerpAPI ===
def format_serpapi_links(resp_json):
    items = resp_json.get("organic_results", [])[:3]
    links = []
    for it in items:
        title, link, snippet = it.get("title"), it.get("link"), it.get("snippet", "")
        if title and link:
            links.append(f"[{title}]({link})\n> {snippet}")
    return "\n\n".join(links)

def web_search(question):
    try:
        params = {"engine": "google", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params).json()
        links = format_serpapi_links(resp)
        if links:
            return f"**Results from Google:**\n\n{links}"
    except:
        pass

    try:
        params = {"engine": "bing", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params).json()
        links = format_serpapi_links(resp)
        if links:
            return f"**Results from Bing:**\n\n{links}"
    except:
        pass

    return "🔎 No useful results found on the web."

# === Setup retrieval-based QA chain ===
@st.cache_resource
def setup_chain(_vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

# === Ask ChatGPT directly if docs fail ===
def ask_chatgpt_fallback(question):
    return ChatOpenAI(temperature=0.7)(question).strip()

# === STREAMLIT APP ===
st.set_page_config(page_title="Nonprofit Chatbot", layout="wide")
st.title("📚 AI Chatbot for Nonprofit Docs + Web")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    with st.spinner("Loading documents and setting up chatbot..."):
        doc_paths = download_google_docs(FOLDER_ID, credentials)
        vector_db = build_vector_store(doc_paths)
        st.session_state.qa_chain = setup_chain(vector_db)

question = st.chat_input("Ask your question...")

fallback_phrases = [
    "i don't have information", "i don't know", "not covered",
    "not available", "best to look up"
]

if question:
    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain({"question": question})
        doc_answer = result.get("answer", "").strip()

        # Step 1: Based on documents
        if not any(phrase in doc_answer.lower() for phrase in fallback_phrases):
            answer = f"📁 Based on docs:\n\n{doc_answer}"
        else:
            # Step 2: Try ChatGPT
            gpt_answer = ask_chatgpt_fallback(question)
            if not any(phrase in gpt_answer.lower() for phrase in fallback_phrases):
                answer = f"💬 Answer generated by ChatGPT:\n\n{gpt_answer}"
            else:
                # Step 3: Web search
                web_answer = web_search(question)
                if "no useful results" not in web_answer.lower():
                    answer = f"🌐 Based on a web search:\n\n{web_answer}"
                else:
                    answer = "❓ Sorry, I couldn’t find any reliable info."

        st.session_state.chat_history.append((question, answer))

# === Show chat history ===
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
