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
import fitz  # PyMuPDF
from langchain.schema import Document

# === CONFIG ===
FOLDER_ID = "1z_zzdbB4zJo70o3rofTqwm30ux9dpRsX"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_KEY

# === Parse and load Google service account credentials from secrets ===
credentials_info = json.loads(st.secrets["GOOGLE_CREDENTIALS_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

def load_pdf_with_fitz(filepath):
    doc = fitz.open(filepath)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    full_text = "\n".join(texts)
    return full_text

@st.cache_resource
def download_google_docs(folder_id, _credentials):
    service = build('drive', 'v3', credentials=_credentials)

    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        pageSize=10,
        fields="files(id, name, mimeType)"
    ).execute()

    files = results.get('files', [])
    
    paths = []
    for file in files:
        if file['mimeType'] == 'application/pdf':
            request = service.files().get_media(fileId=file['id'])
            filepath = f"/tmp/{file['name']}"
            with open(filepath, 'wb') as f:
                f.write(request.execute())
            paths.append(filepath)

    return paths

@st.cache_resource
def build_vector_store(docs_paths):
    documents = []
    for path in docs_paths:
        text = load_pdf_with_fitz(path)
        documents.append(Document(page_content=text, metadata={"source": path}))
    return Chroma.from_documents(documents, OpenAIEmbeddings())

def web_search(question):
    params = {
        "engine": "google",
        "q": question,
        "api_key": SERPAPI_KEY
    }
    response = requests.get("https://serpapi.com/search", params=params).json()
    results = response.get("organic_results", [])
    
    if not results:
        return "🔎 No useful results found on the web."

    # Build markdown-style link list
    links = []
    for res in results[:3]:  # Limit to top 3
        title = res.get("title", "Link")
        link = res.get("link", "#")
        snippet = res.get("snippet", "")
        links.append(f"[{title}]({link})\n> {snippet}")

    return "\n\n".join(links)


@st.cache_resource
def setup_chain(_vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # explicitly tell which output to store
    )
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return qa_chain

# === STREAMLIT UI ===
st.set_page_config(page_title="Nonprofit Chatbot", layout="wide")
st.title("📚 AI Chatbot for Nonprofit Docs + Web")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    with st.spinner("Loading documents and setting up chatbot..."):
        docs = download_google_docs(FOLDER_ID, credentials)
        vectordb = build_vector_store(docs)
        st.session_state.qa_chain = setup_chain(vectordb)

question = st.chat_input("Ask your question...")

if question:
    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain({"question": question})
        answer = result.get("answer", "").strip()
        
        # Simple fallback trigger check
        fallback_phrases = [
            "i don't have information", "not available in the provided context",
            "i don't know", "i recommend checking", "unfortunately"
        ]
        fallback = any(p in answer.lower() for p in fallback_phrases) or len(answer) < 20

        if fallback:
            web_answer = web_search(question)
            answer += f"\n\n---\n🔎 **Based on a web search:**\n\n{web_answer}"

        st.session_state.chat_history.append((question, answer))


# Show history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

