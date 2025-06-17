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
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Google API components
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

# PDF processing
import fitz  # PyMuPDF

# === CONFIGURATION & SECRETS ===
# It's recommended to set your Google Drive folder ID here
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
    
    # Debug info
    total_tokens = sum(estimate_tokens(doc.page_content) for doc in chunked_docs)
    print(f"DEBUG: Chunked into {len(chunked_docs)} documents, total estimated tokens: {total_tokens}")
    return chunked_docs

def web_search(question):
    """Performs a web search using SerpApi and returns formatted results."""
    try:
        params = {
            "engine": "google",
            "q": question,
            "api_key": SERPAPI_KEY
        }
        # Increased timeout for more reliability
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        
        # This line will raise an error for bad responses (like 4xx or 5xx)
        resp.raise_for_status() 
        
        data = resp.json()
        results = data.get("organic_results", [])[:3]
        
        if "error" in data:
            # SerpApi often returns a 200 OK but with an error message in the JSON
            return f"Web search failed: SerpApi returned an error - {data.get('error')}"

        output = []
        for r in results:
            title = r.get("title")
            link = r.get("link")
            snippet = r.get("snippet", "")
            if title and link:
                output.append(f"[{title}]({link})\n> {snippet}")
        
        return "\n\n".join(output) if output else "No results found."

    except requests.exceptions.Timeout:
        print("Web search failed: The request timed out.")
        return "Web search failed: The request to SerpApi timed out."
    except requests.exceptions.RequestException as e:
        # This will catch most other network-related errors
        print(f"Web search failed: {e}")
        return f"Web search failed: A network error occurred. **Details:** {e}"

# === LANGCHAIN SETUP ===

@st.cache_resource
def build_vectorstore(_documents):
    """Builds a Chroma vector store from documents."""
    return Chroma.from_documents(_documents, OpenAIEmbeddings())

@st.cache_resource
def setup_chain(_vectorstore):
    """Sets up the ConversationalRetrievalChain with custom prompts for robust fallback."""
    # 1. Condense Question Prompt: Merges chat history and a new question into a standalone question.
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # 2. QA Prompt: The main prompt for answering, with a specific instruction for when the answer is not found.
    qa_template = """You are an expert assistant for answering questions based on provided documents.
Your goal is to provide accurate and concise answers from the given context.
Do not make up information. Only use the context provided below.

Context:
{context}

Question:
{question}

---
Instructions:
- Analyze the context and the question carefully.
- If the context contains the answer, provide it directly.
- If the context does not contain enough information to answer the question, you MUST respond with the single word 'NO_ANSWER' and nothing else.
- Do not add any pleasantries or introductory phrases to your answer.

Answer:
"""
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(qa_template)

    # 3. Initialize LLM, Memory, and Retriever
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    memory = ConversationBufferWindowMemory(
        k=3,
        return_messages=True,
        memory_key="chat_history",
        output_key='answer'
    )
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Create the chain with custom prompts
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT},
        return_source_documents=True
    )
    return chain

# === STREAMLIT UI ===

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 AI Chatbot with Web Search Fallback")
st.info("This chatbot answers questions based on documents from a Google Drive folder. If it can't find an answer in the documents, it will automatically search the web.")

# Initialize session state for chat history and the chain
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# On first run, load documents and set up the chain
if st.session_state.qa_chain is None:
    with st.spinner("Initializing: Loading documents from Google Drive, chunking, and building vector store..."):
        if FOLDER_ID == "YOUR_GOOGLE_DRIVE_FOLDER_ID":
            st.warning("Please replace 'YOUR_GOOGLE_DRIVE_FOLDER_ID' with your actual Google Drive folder ID in the code.", icon="⚠️")
            st.stop()
        
        docs = download_and_process_files(FOLDER_ID, DRIVE_SERVICE)
        if not docs:
            st.error("No documents were found or processed from the specified Google Drive folder. Please check the folder ID and file permissions.")
            st.stop()

        chunked_docs = chunk_documents(docs)
        vectordb = build_vectorstore(chunked_docs)
        st.session_state.qa_chain = setup_chain(vectordb)
        st.success("Initialization complete. You can now ask questions.")


# Display previous chat messages
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Handle new user input
if question := st.chat_input("Ask a question about the documents..."):
    if st.session_state.qa_chain is None:
        st.error("The chatbot is not initialized. Please refresh the page.")
        st.stop()
        
    st.session_state.chat_history.append((question, ""))
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            message_placeholder = st.empty()
            
            # Call the chain
            result = st.session_state.qa_chain({"question": question})
            answer = result["answer"].strip()

            # Implement the robust fallback logic
            if answer == "NO_ANSWER":
                message_placeholder.markdown("Couldn't find an answer in the documents. Searching the web...")
                web_answer = web_search(question)
                final_answer = f"I couldn't find a specific answer in the provided documents. A web search suggests the following:\n\n---\n\n{web_answer}"
            else:
                final_answer = answer

            message_placeholder.markdown(final_answer)

    # Append the final answer to the full chat history
    st.session_state.chat_history[-1] = (question, final_answer)