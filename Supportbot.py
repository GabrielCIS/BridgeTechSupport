import os
import json
import io
import csv
import sys
import streamlit as st
import tiktoken


import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter



from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

import requests
import fitz  # PyMuPDF

# === CONFIG ===
FOLDER_ID = "1z_zzdbB4zJo70o3rofTqwm30ux9dpRsX"

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_KEY"] = SERPAPI_KEY

# === GOOGLE AUTH ===
credentials_info = json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

# ==== DEBUG ===
enc = tiktoken.encoding_for_model("gpt-4")  # or "gpt-3.5-turbo"

def estimate_tokens(text):
    return len(enc.encode(text))

# === HELPERS ===
def load_pdf(filepath):
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

def download_files(folder_id, mime_types, service):
    files = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        pageSize=50,
        fields="files(id, name, mimeType)"
    ).execute().get('files', [])

    documents = []
    for file in files:
        if file['mimeType'] not in mime_types:
            continue

        file_id = file['id']
        file_name = file['name']
        mime = file['mimeType']

        if mime == "application/pdf":
            request = service.files().get_media(fileId=file_id)
            with open(f"/tmp/{file_name}", "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            text = load_pdf(f"/tmp/{file_name}")
            documents.append(Document(page_content=text, metadata={"source": file_name}))

        elif mime == "application/vnd.google-apps.document":
            request = service.files().export_media(fileId=file_id, mimeType="text/plain")
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            text = fh.getvalue().decode("utf-8")
            documents.append(Document(page_content=text, metadata={"source": file_name}))

        elif mime == "application/vnd.google-apps.spreadsheet":
            request = service.files().export_media(fileId=file_id, mimeType="text/csv")
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            csv_text = fh.getvalue().decode("utf-8")
            rows = csv.reader(csv_text.splitlines())
            flat_text = "\n".join([", ".join(row) for row in rows])
            documents.append(Document(page_content=flat_text, metadata={"source": file_name}))

        elif mime == "text/html":
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            html_text = fh.getvalue().decode("utf-8")
            documents.append(Document(page_content=html_text, metadata={"source": file_name}))

    return documents

@st.cache_resource
def build_vectorstore(_documents):
    return Chroma.from_documents(_documents, OpenAIEmbeddings())

def chunk_documents_orig(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for chunk in splits:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunked_docs

# DEBUG: Add token estimation and logging
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for chunk in splits:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    # Debug info
    total_tokens = sum(estimate_tokens(doc.page_content) for doc in chunked_docs)
    print(f"DEBUG: Chunked into {len(chunked_docs)} documents, total estimated tokens: {total_tokens}")
    for i, doc in enumerate(chunked_docs[:5]):  # print first 5 chunks info
        print(f"DEBUG: Chunk {i} tokens: {estimate_tokens(doc.page_content)}, source: {doc.metadata.get('source', 'unknown')}")
    return chunked_docs

def web_search(question):
    try:
        params = {"engine": "google", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params).json()
        results = resp.get("organic_results", [])[:3]
        output = []
        for r in results:
            title = r.get("title")
            link = r.get("link")
            snippet = r.get("snippet", "")
            if title and link:
                output.append(f"[{title}]({link})\n> {snippet}")
        return "\n\n".join(output) if output else "No results found."
    except:
        return "Web search failed."

@st.cache_resource


@st.cache_resource
def setup_chain(_vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferWindowMemory(
        k=3,  # only keep last 3 interactions in memory
        return_messages=True,
        memory_key="chat_history"
    )
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 2})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain

# === Streamlit UI ===
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 AI Chatbot with Token Safety 🚦")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    with st.spinner("Loading..."):
        service = build('drive', 'v3', credentials=credentials)
        docs = []
        for mt in ["application/pdf", "application/vnd.google-apps.document",
                   "application/vnd.google-apps.spreadsheet", "text/html"]:
            docs.extend(download_files(FOLDER_ID, [mt], service))

        chunked_docs = chunk_documents(docs)  # <-- only once after the loop

        vectordb = build_vectorstore(chunked_docs)
        st.session_state.qa_chain = setup_chain(vectordb)
        


question = st.chat_input("Ask something...")


if question:
    with st.spinner("Thinking..."):
        # Get retriever results explicitly for debug
        retriever = st.session_state.qa_chain.retriever
        docs = retriever.get_relevant_documents(question)
        print(f"DEBUG: Retriever returned {len(docs)} docs")
        total_tokens_docs = sum(estimate_tokens(doc.page_content) for doc in docs)
        print(f"DEBUG: Retriever docs total tokens: {total_tokens_docs}")

        result = st.session_state.qa_chain({"question": question})

        answer = result["answer"]

        fallback_phrases = ["don't know", "not available", "no information"]
        if any(p in answer.lower() for p in fallback_phrases):
            web_answer = web_search(question)
            answer = f"🌐 Web answer:\n\n{web_answer}"

        # Append to local chat_history to show chat UI
        st.session_state.chat_history.append((question, answer))

        # Periodically clear chat history and memory to avoid token overflow
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = []
            st.session_state.qa_chain.memory.clear()

# Render last 5 messages in chat UI
for q, a in st.session_state.chat_history[-5:]:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)