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

def web_search_orig(question):
    # 1️⃣ Try SerpAPI Google
    try:
        params = {"engine": "google", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params).json()
        links = format_serpapi_links(resp)
        if links:
            return links
    except:
        pass

    # 2️⃣ Try SerpAPI Bing
    try:
        params = {"engine": "bing", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params).json()
        links = format_serpapi_links(resp)
        if links:
            return links
    except:
        pass

    # 3️⃣ Try Brave Search API (example point)
    try:
        resp = requests.get(f"https://api.duckduckgo.com/?q={question}&format=json").json()
        related = resp.get("RelatedTopics", [])[:3]
        links = [f"- [{r.get('Text')}]({r.get('FirstURL')})" for r in related if r.get("FirstURL")]
        if links:
            return "\n".join(links)
    except:
        pass

    return "🔎 No useful results found on the web."

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


def format_serpapi_links(resp_json):
    items = resp_json.get("organic_results", [])[:3]
    links = []
    for it in items:
        title, link, snippet = it.get("title"), it.get("link"), it.get("snippet", "")
        if title and link:
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
def ask_chatgpt_fallback(question):
    response = openai_chat.invoke(question)
    return response.content.strip()

if question:
    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain({"question": question})
        answer = result["answer"]
        label = "📁 Based on your uploaded documents."

        # If vector store didn’t help
        if not result.get("source_documents") or "i don't know" in answer.lower():
            # Try ChatGPT
            gpt_response = ask_chatgpt_fallback(question)
            if gpt_response and "I don't know" not in gpt_response.lower():
                answer = gpt_response
                label = "💬 Answer generated by ChatGPT."
            else:
                # Fallback to web search
                web_answer = web_search(question)
                if web_answer and "no useful results" not in web_answer.lower():
                    answer = web_answer
                    label = "🌐 Based on a web search."
                else:
                    answer = "❓ Sorry, I couldn’t find any useful information."
                    label = ""

        st.session_state.chat_history.append((question, f"{label}\n\n{answer}"))



# Show history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

