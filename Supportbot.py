import os
import json
import io
import csv
import streamlit as st
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.schema import Document

from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

import requests
import fitz  # PyMuPDF for PDF parsing

# === CONFIGURATION ===
# Replace these with your actual folder IDs or set in Streamlit secrets
Single_FOLDER_ID = "1z_zzdbB4zJo70o3rofTqwm30ux9dpRsX"  # For single file retrieval
PDF_FOLDER_ID = Single_FOLDER_ID
DOCS_FOLDER_ID = Single_FOLDER_ID
SHEETS_FOLDER_ID = Single_FOLDER_ID
CLASSIC_SITES_FOLDER_ID = Single_FOLDER_ID

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_KEY"] = SERPAPI_KEY

# Load Google credentials
credentials_info = json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

# --- Helper to download PDFs ---
def load_pdf_with_fitz(filepath):
    doc = fitz.open(filepath)
    texts = [page.get_text() for page in doc]
    return "\n".join(texts)

# --- Download files from Drive folder ---
def download_files_from_folder(folder_id, mime_types, service):
    """
    Downloads files of specified mime_types from the given Drive folder.
    Returns list of Document objects.
    """
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        pageSize=50,
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get('files', [])

    documents = []

    for file in files:
        if file['mimeType'] not in mime_types:
            continue

        file_id = file['id']
        file_name = file['name']
        mime = file['mimeType']

        if mime == "application/pdf":
            # Download PDF binary
            request = service.files().get_media(fileId=file_id)
            filepath = f"/tmp/{file_name}"
            with open(filepath, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            text = load_pdf_with_fitz(filepath)
            documents.append(Document(page_content=text, metadata={"source": file_name}))

        elif mime == "application/vnd.google-apps.document":
            # Export Google Doc as plain text
            request = service.files().export_media(fileId=file_id, mimeType="text/plain")
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            text = fh.read().decode("utf-8")
            documents.append(Document(page_content=text, metadata={"source": file_name}))

        elif mime == "application/vnd.google-apps.spreadsheet":
            # Export Google Sheet as CSV text
            request = service.files().export_media(fileId=file_id, mimeType="text/csv")
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            csv_text = fh.read().decode("utf-8")

            # Optional: flatten CSV into plain text for indexing
            reader = csv.reader(csv_text.splitlines())
            rows = list(reader)
            flat_text = "\n".join([", ".join(row) for row in rows])

            documents.append(Document(page_content=flat_text, metadata={"source": file_name}))

        elif mime == "text/html":
            # Classic Sites pages might be stored as HTML files
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            html_text = fh.read().decode("utf-8")
            documents.append(Document(page_content=html_text, metadata={"source": file_name}))

    return documents

# === Build vector store from all documents ===
@st.cache_resource
def build_vector_store(_all_documents):  # <-- underscore added here
    return Chroma.from_documents(_all_documents, OpenAIEmbeddings())

# === Format SERPAPI results for web search fallback ===
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
    except Exception as e:
        st.error(f"Google search failed: {e}")

    try:
        params = {"engine": "bing", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params).json()
        links = format_serpapi_links(resp)
        if links:
            return f"**Results from Bing:**\n\n{links}"
    except Exception as e:
        st.error(f"Bing search failed: {e}")

    return "🔎 No useful results found on the web."

# === Setup Conversational Retrieval Chain with Summarizing Memory ===
@st.cache_resource
def setup_chain(_vectorstore):  # <-- underscore added here
    llm = ChatOpenAI(temperature=0)
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain

# === Fallback to ChatGPT if docs retrieval fails ===
def ask_chatgpt_fallback(question):
    resp = ChatOpenAI(temperature=0.7)(question)
    return resp.strip()

# === MAIN STREAMLIT APP ===
st.set_page_config(page_title="Nonprofit Chatbot with Google Docs, Sheets & Sites", layout="wide")
st.title("📚 AI Chatbot: PDFs, Docs, Sheets & Classic Sites")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    with st.spinner("Loading documents and setting up chatbot..."):
        service = build('drive', 'v3', credentials=credentials)

        # Download and parse all docs from all sources
        docs_pdf = download_files_from_folder(PDF_FOLDER_ID, ["application/pdf"], service)
        docs_docs = download_files_from_folder(DOCS_FOLDER_ID, ["application/vnd.google-apps.document"], service)
        docs_sheets = download_files_from_folder(SHEETS_FOLDER_ID, ["application/vnd.google-apps.spreadsheet"], service)
        docs_sites = download_files_from_folder(CLASSIC_SITES_FOLDER_ID, ["text/html"], service)

        # Combine all docs
        all_docs = docs_pdf + docs_docs + docs_sheets + docs_sites

        vectordb = build_vector_store(all_docs)
        st.session_state.qa_chain = setup_chain(vectordb)

question = st.chat_input("Ask your question...")

fallback_phrases = [
    "i don't have information", "i don't know",
    "not covered", "not available", "best to look up"
]

if question:
    with st.spinner("Thinking..."):
        # Step 1: Docs-based answer
        result = st.session_state.qa_chain({"question": question})
        doc_answer = result.get("answer", "").strip()

        use_gpt = any(phrase in doc_answer.lower() for phrase in fallback_phrases)
        if not use_gpt:
            answer = f"📁 Based on docs:\n\n{doc_answer}"
        else:
            # Step 2: fallback to ChatGPT
            gpt_answer = ask_chatgpt_fallback(question)
            if not any(phrase in gpt_answer.lower() for phrase in fallback_phrases):
                answer = f"💬 Answer generated by ChatGPT:\n\n{gpt_answer}"
            else:
                # Step 3: fallback to web search
                web_answer = web_search(question)
                if "no useful results" not in web_answer.lower():
                    answer = f"🌐 Based on a web search:\n\n{web_answer}"
                else:
                    answer = "❓ Sorry, I couldn’t find any reliable info."

        st.session_state.chat_history.append((question, answer))

# Display chat history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
