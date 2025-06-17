import os
import json
import io
import csv
import sys
import streamlit as st
import tiktoken
import requests

# Use a specific version of pysqlite3 to avoid potential conflicts
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# LangChain components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Google API components
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

# PDF processing
import fitz  # PyMuPDF

# === CONFIGURATION & SECRETS ===
# IMPORTANT: Replace with your actual Google Drive folder ID
FOLDER_ID = "1z_zzdbB4zJo70o3rofTqwm30ux9dpRsX" 

# Load secrets from Streamlit's secrets management
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
    GOOGLE_CREDENTIALS_JSON = st.secrets["GOOGLE_CREDENTIALS_JSON"]
except FileNotFoundError:
    st.error("Secrets file not found. Please ensure you have a .streamlit/secrets.toml file with your API keys.")
    st.stop()
except KeyError as e:
    st.error(f"Missing secret: {e}. Please check your .streamlit/secrets.toml file.")
    st.stop()


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_KEY"] = SERPAPI_KEY

# === GOOGLE AUTHENTICATION ===
try:
    credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    DRIVE_SERVICE = build('drive', 'v3', credentials=credentials)
except json.JSONDecodeError:
    st.error("Failed to parse Google credentials. Please check the format of your GOOGLE_CREDENTIALS_JSON secret.")
    st.stop()


# === DEBUGGING & TOKEN ESTIMATION ===
enc = tiktoken.encoding_for_model("gpt-4")

def estimate_tokens(text):
    """Estimates the number of tokens in a given text."""
    return len(enc.encode(text))

# === HELPER FUNCTIONS ===

def load_pdf_text(filepath):
    """Extracts text from a PDF file."""
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

def download_and_process_files(folder_id, service):
    """Downloads and processes files from a Google Drive folder into LangChain Documents."""
    mime_types_map = {
        "application/pdf": "application/pdf",
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
        "text/html": "text/html"
    }

    query = f"'{folder_id}' in parents and trashed = false"
    files = service.files().list(q=query, pageSize=50, fields="files(id, name, mimeType)").execute().get('files', [])

    documents = []
    st.write(f"Found {len(files)} files in Google Drive folder.")

    for file in files:
        file_id = file['id']
        file_name = file['name']
        mime_type = file['mimeType']
        
        try:
            if mime_type == "application/pdf":
                request = service.files().get_media(fileId=file_id)
                with open(f"/tmp/{file_name}", "wb") as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                text = load_pdf_text(f"/tmp/{file_name}")
                documents.append(Document(page_content=text, metadata={"source": file_name}))
            elif mime_type in ["application/vnd.google-apps.document", "application/vnd.google-apps.spreadsheet", "text/html"]:
                export_mime_type = mime_types_map[mime_type]
                request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                
                text = fh.getvalue().decode("utf-8", errors='ignore')
                if export_mime_type == "text/csv":
                    rows = csv.reader(text.splitlines())
                    text = "\n".join([", ".join(row) for row in rows])
                
                documents.append(Document(page_content=text, metadata={"source": file_name}))
        except Exception as e:
            st.warning(f"Could not process file: {file_name} (ID: {file_id}). Reason: {e}")

    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Splits documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = text_splitter.split_documents(documents)
    total_tokens = sum(estimate_tokens(doc.page_content) for doc in chunked_docs)
    print(f"DEBUG: Chunked into {len(chunked_docs)} documents, total estimated tokens: {total_tokens}")
    return chunked_docs

# === KNOWLEDGE SOURCE FUNCTIONS ===

def web_search(question):
    """Performs a web search using SerpApi and returns formatted results."""
    try:
        params = {"engine": "google", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "error" in data:
            return f"Web search failed: SerpApi returned an error - {data.get('error')}"

        results = data.get("organic_results", [])[:3]
        output = [f"[{r.get('title')}]({r.get('link')})\n> {r.get('snippet', '')}" for r in results if r.get('title') and r.get('link')]
        return "\n\n".join(output) if output else "No results found."

    except requests.exceptions.Timeout:
        return "Web search failed: The request to SerpApi timed out."
    except requests.exceptions.RequestException as e:
        return f"Web search failed: A network error occurred. Details: {e}"

def ask_chatgpt(question):
    """Gets a direct answer from a general-purpose OpenAI model (the 'ChatGPT' source)."""
    print("DEBUG: Asking general knowledge source (ChatGPT).")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant