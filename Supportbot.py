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
        ("system", "You are a helpful assistant. Answer the user's question clearly and concisely."),
        ("human", "{question}")
    ])
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# === LANGCHAIN SETUP FOR RAG (Primary Source) ===

@st.cache_resource
def build_vectorstore(_documents):
    """Builds a Chroma vector store from documents."""
    return Chroma.from_documents(_documents, OpenAIEmbeddings())

@st.cache_resource
def setup_rag_chain(_vectorstore):
    """Sets up the ConversationalRetrievalChain for document-based Q&A (RAG)."""
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History: {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    qa_template = """You are an expert assistant for answering questions based on provided documents.
    Your goal is to provide accurate answers from the given context. Only use the context provided.
    Context: {context}
    Question: {question}
    Instructions: If the context does not contain enough information, you MUST respond with the single word 'NO_ANSWER' and nothing else.
    Answer:"""
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(qa_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="chat_history", output_key='answer')
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

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

st.set_page_config(page_title="Tiered RAG Chatbot", layout="wide")
st.title("📚🤖🌐 Tiered AI Chatbot")
st.info("This bot answers in three stages: 1. Your Documents, 2. AI Assistant (ChatGPT), 3. Web Search.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if st.session_state.rag_chain is None:
    with st.spinner("Initializing: Loading documents and setting up the RAG chain..."):
        if FOLDER_ID == "YOUR_GOOGLE_DRIVE_FOLDER_ID": # Replace with your actual ID
            st.warning("Please replace 'YOUR_GOOGLE_DRIVE_FOLDER_ID' with your actual Google Drive folder ID in the code.", icon="⚠️")
            st.stop()
        
        docs = download_and_process_files(FOLDER_ID, DRIVE_SERVICE)
        if not docs:
            st.error("No documents were found or processed. Please check the folder ID and file permissions.")
            st.stop()

        chunked_docs = chunk_documents(docs)
        vectordb = build_vectorstore(chunked_docs)
        st.session_state.rag_chain = setup_rag_chain(vectordb)
        st.success("Initialization complete.")

# Display previous chat messages
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Handle new user input
if question := st.chat_input("Ask a question..."):
    if st.session_state.rag_chain is None:
        st.error("The chatbot is not initialized. Please refresh the page.")
        st.stop()
        
    st.session_state.chat_history.append((question, ""))
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Checking documents..."):
            message_placeholder = st.empty()
            
            # --- TIER 1: RAG (Your Documents) ---
            rag_result = st.session_state.rag_chain({"question": question})
            rag_answer = rag_result["answer"].strip()

            if rag_answer != "NO_ANSWER":
                final_answer = f"📄 **From Documents:**\n\n{rag_answer}"
                message_placeholder.markdown(final_answer)
            else:
                # --- TIER 2: ChatGPT (General Knowledge) ---
                message_placeholder.markdown("No answer in documents. Asking AI assistant...")
                with st.spinner("Asking AI assistant..."):
                    chatgpt_answer = ask_chatgpt(question)
                    
                    # Simple check to see if the AI model can't answer
                    # Models are often trained to mention their knowledge cutoff date
                    fallback_triggers = [
                        "as of my last update", "my knowledge cutoff", "i don't have real-time", 
                        "i cannot access the internet", "i am unable to provide"
                    ]
                    
                    if not any(trigger in chatgpt_answer.lower() for trigger in fallback_triggers):
                        final_answer = f"🤖 **From AI Assistant:**\n\n{chatgpt_answer}"
                        message_placeholder.markdown(final_answer)
                    else:
                        # --- TIER 3: Web Search ---
                        message_placeholder.markdown("AI assistant has limited info. Searching the web...")
                        with st.spinner("Searching the web..."):
                            web_answer = web_search(question)
                            final_answer = f"🌐 **From Web Search:**\n\n{web_answer}"
                            message_placeholder.markdown(final_answer)

    # Append the final answer to the full chat history
    st.session_state.chat_history[-1] = (question, final_answer)


    # === DEBUGGING SECTION ===
st.divider()
with st.expander("🛠️ Click here for advanced debugging tools"):
    st.subheader("Retriever Debugger")
    st.warning("This tool allows you to see exactly which text chunks are being retrieved for a given question.", icon="🔬")

    debug_question = st.text_input("Enter a question to test the retriever:")

    if st.button("Test Retriever"):
        if st.session_state.rag_chain and debug_question:
            retriever = st.session_state.rag_chain.retriever
            with st.spinner("Finding relevant documents..."):
                try:
                    retrieved_docs = retriever.get_relevant_documents(debug_question)
                    st.info(f"Retrieved **{len(retrieved_docs)}** chunks for your question.")

                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"--- \n**Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')})**")
                        # Using a code block to preserve formatting and show the raw text
                        st.code(doc.page_content, language=None)

                except Exception as e:
                    st.error(f"An error occurred during retrieval: {e}")
        elif not debug_question:
            st.warning("Please enter a question to test.")
        else:
            st.error("The RAG chain is not initialized.")