import os
import json
import io
import csv
import sys
import streamlit as st
import tiktoken
import requests
import base64

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
from langchain_core.messages import HumanMessage

# Google API components
import google.generativeai as genai
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
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"Missing a secret key in secrets.toml: {e}")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPAPI_KEY"] = SERPAPI_KEY

# === API INITIALIZATION ===
try:
    # Google Drive
    credentials_info = json.loads(GOOGLE_CREDENTIALS_JSON)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    DRIVE_SERVICE = build('drive', 'v3', credentials=credentials)
    
    # Google Gemini
    genai.configure(api_key=GEMINI_API_KEY)

except Exception as e:
    st.error(f"Failed to initialize an API client. Please check your credentials. Error: {e}")
    st.stop()

# === DEBUGGING & TOKEN ESTIMATION ===
enc = tiktoken.encoding_for_model("gpt-4")
def estimate_tokens(text):
    return len(enc.encode(text))

# === HELPER & DATA PROCESSING FUNCTIONS ===

def load_pdf_text(filepath):
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

def download_and_process_files(folder_id, service):
    text_documents = []
    image_files = [] 
    
    query = f"'{folder_id}' in parents and trashed = false"
    files = service.files().list(q=query, pageSize=100, fields="files(id, name, mimeType)").execute().get('files', [])
    st.write(f"Found {len(files)} total items in Google Drive folder.")

    text_mime_types = {"application/pdf": "application/pdf", "application/vnd.google-apps.document": "text/plain", "application/vnd.google-apps.spreadsheet": "text/csv", "text/html": "text/html"}
    image_mime_types = ["image/jpeg", "image/png"]

    for file in files:
        file_id, file_name, mime_type = file['id'], file['name'], file['mimeType']
        try:
            if mime_type in text_mime_types:
                request = service.files().export_media(fileId=file_id, mimeType=text_mime_types[mime_type]) if "google-apps" in mime_type else service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                MediaIoBaseDownload(fh, request).next_chunk()
                text = fh.getvalue().decode("utf-8", errors='ignore')
                if text_mime_types[mime_type] == "text/csv":
                    rows = csv.reader(text.splitlines())
                    text = "\n".join([", ".join(row) for row in rows])
                text_documents.append(Document(page_content=text, metadata={"source": file_name}))
            elif mime_type in image_mime_types:
                image_files.append({'id': file_id, 'name': file_name})
        except Exception as e:
            st.warning(f"Could not process file: {file_name}. Reason: {e}")

    return text_documents, image_files

def generate_caption_for_image(file_id, service):
    print(f"DEBUG: Generating caption for image ID: {file_id} using Gemini")
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        MediaIoBaseDownload(fh, request).next_chunk()
        image_data = fh.getvalue()
        model = genai.GenerativeModel('gemini-pro-vision')
        prompt_text = "Describe this architectural image or technical diagram in detail. Mention the style, materials, key components, connections, and overall impression. This description will be used to search for the image later."
        contents = [prompt_text, {"mime_type": "image/jpeg", "data": image_data}]
        response = model.generate_content(contents)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"ERROR: Failed to generate caption for image {file_id} using Gemini. Reason: {e}")
        return None

def chunk_documents(documents, chunk_size=1500, chunk_overlap=500):
    """Chunking with parameters found to be effective during debugging."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# === KNOWLEDGE SOURCE & LANGCHAIN FUNCTIONS ===

def web_search(question):
    try:
        params = {"engine": "google", "q": question, "api_key": SERPAPI_KEY}
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status(); data = resp.json()
        if "error" in data: return f"Web search failed: {data.get('error')}"
        results = data.get("organic_results", [])[:3]
        output = [f"[{r.get('title')}]({r.get('link')})\n> {r.get('snippet', '')}" for r in results if r.get('title') and r.get('link')]
        return "\n\n".join(output) if output else "No results found."
    except Exception as e: return f"Web search failed: {e}"

def ask_chatgpt(question):
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
    return llm.invoke(f"You are a helpful assistant. Answer the user's question clearly and concisely.\n\nQuestion: {question}").content

@st.cache_resource
def build_vectorstore(_documents):
    return Chroma.from_documents(_documents, OpenAIEmbeddings())

@st.cache_resource
def setup_rag_chain(_vectorstore):
    _template = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question... Chat History: {chat_history}\nFollow Up Input: {question}\nStandalone question:"
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    # --- "Softened" Prompt ---
    # This version is less strict, giving the LLM more flexibility to answer
    # if the context is relevant but not a perfect 100% match.
    qa_template = """You are an expert assistant for answering questions based on provided documents.
    Your goal is to provide accurate answers from the given context. Use only the context provided below.

    Context:
    {context}

    Question:
    {question}
    
    Instructions:
    Based only on the context provided, answer the question.
    If the context does not contain the answer, simply say that you could not find the answer in the provided documents. Do not make up an answer.

    Answer:
    """
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(qa_template)
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="chat_history", output_key='answer')
    
    # --- Key Tuning Parameter `k` ---
    # This value was increased during debugging. It's a key parameter to adjust.
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 10})
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        condense_question_prompt=CONDENSE_QUESTION_PROMPT, 
        combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT}, 
        return_source_documents=True
    )

# === STREAMLIT UI ===
st.set_page_config(page_title="Multimodal RAG Chatbot", layout="wide")
st.title("🏛️🤖🖼️ Multimodal Architecture Chatbot")
st.info("This bot analyzes documents and images from a Google Drive folder using RAG and Gemini Vision.")

# Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if st.session_state.rag_chain is None:
    with st.spinner("Initializing: This may take a while for new images..."):
        if FOLDER_ID == "YOUR_GOOGLE_DRIVE_FOLDER_ID":
            st.warning("Please update the FOLDER_ID in the script.", icon="⚠️"); st.stop()
        
        st.write("Step 1/4: Loading files from Google Drive...")
        text_docs, image_files = download_and_process_files(FOLDER_ID, DRIVE_SERVICE)
        
        st.write(f"Step 2/4: Found {len(image_files)} images. Generating captions using Gemini Vision...")
        caption_docs = [Document(page_content=caption, metadata={'source': img['name'], 'is_image_caption': True, 'image_file_id': img['id']}) for img in image_files if (caption := generate_caption_for_image(img['id'], DRIVE_SERVICE))]
        
        st.write("Step 3/4: Combining and chunking all content...")
        all_docs = text_docs + caption_docs
        if not all_docs: st.error("No documents or images could be processed."); st.stop()
        chunked_docs = chunk_documents(all_docs)
        
        st.write("Step 4/4: Building vector store and initializing RAG chain...")
        vectordb = build_vectorstore(chunked_docs)
        st.session_state.rag_chain = setup_rag_chain(vectordb)
        st.success("Initialization complete.")

# Display chat history
for q, a in st.session_state.chat_history:
    with st.chat_message("user"): st.write(q)
    with st.chat_message("assistant"): st.write(a)

# Handle new user input
if question := st.chat_input("Ask about your documents or diagrams..."):
    st.session_state.chat_history.append((question, ""))
    with st.chat_message("user"): st.write(question)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            rag_result = st.session_state.rag_chain({"question": question})
            
            source_is_image, retrieved_image_id = False, None
            if rag_result.get("source_documents"):
                for doc in rag_result["source_documents"]:
                    if doc.metadata.get('is_image_caption'):
                        source_is_image, retrieved_image_id = True, doc.metadata['image_file_id']
                        break
            
            if source_is_image and retrieved_image_id:
                message_placeholder.markdown("Found a relevant image. Analyzing with Gemini Vision...")
                try:
                    request = DRIVE_SERVICE.files().get_media(fileId=retrieved_image_id)
                    fh = io.BytesIO()
                    MediaIoBaseDownload(fh, request).next_chunk()
                    image_data = fh.getvalue()
                    model = genai.GenerativeModel('gemini-pro-vision')
                    contents = [question, {"mime_type": "image/jpeg", "data": image_data}]
                    response = model.generate_content(contents); response.resolve()
                    final_answer = f"🖼️ **From Image Analysis (Gemini):**\n\n{response.text}"
                except Exception as e: final_answer = f"Sorry, I failed to analyze the image. Error: {e}"
            elif "could not find the answer" not in rag_result["answer"]:
                final_answer = f"📄 **From Documents:**\n\n{rag_result['answer']}"
            else:
                message_placeholder.markdown("No answer in documents. Asking AI assistant...")
                chatgpt_answer = ask_chatgpt(question)
                if not any(trigger in chatgpt_answer.lower() for trigger in ["as of my last update", "my knowledge cutoff"]):
                    final_answer = f"🤖 **From AI Assistant:**\n\n{chatgpt_answer}"
                else:
                    message_placeholder.markdown("AI assistant has limited info. Searching the web...")
                    web_answer = web_search(question)
                    final_answer = f"🌐 **From Web Search:**\n\n{web_answer}"
            
            message_placeholder.markdown(final_answer)
    st.session_state.chat_history[-1] = (question, final_answer)

# === COMPLETE DEBUGGING SUITE ===
st.divider()
with st.expander("🛠️ Click here for advanced RAG debugging tools"):
    st.subheader("Retriever Debugger")
    st.warning("This tool shows you the raw chunks retrieved from the vector store for a given question.", icon="🔬")
    debug_question = st.text_input("Enter a question to test the retriever:", key="debug_question")

    if st.button("Test Retriever", key="debug_button"):
        if st.session_state.rag_chain and debug_question:
            retriever = st.session_state.rag_chain.retriever
            with st.spinner("Finding relevant documents..."):
                retrieved_docs = retriever.get_relevant_documents(debug_question)
                st.info(f"Retrieved **{len(retrieved_docs)}** chunks for your question.")
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"--- \n**Chunk {i+1}** | **Source:** `{doc.metadata.get('source', 'Unknown')}`")
                    if doc.metadata.get('is_image_caption'): st.markdown(f"**Type:** Image Caption | **Image ID:** `{doc.metadata.get('image_file_id', 'N/A')}`")
                    st.code(doc.page_content, language=None)

    st.subheader("Generation Debugger")
    st.markdown("Use this to test the final LLM call with a specific piece of context.")
    if st.session_state.rag_chain:
        qa_prompt = st.session_state.rag_chain.combine_docs_chain.llm_chain.prompt
        context_to_test = st.text_area("Paste the exact content of a single chunk here:", height=250)
        question_to_test = st.text_input("Enter the same question again:", key="generation_debug_question")
        if st.button("Test Generation", key="generation_debug_button"):
            if context_to_test and question_to_test:
                with st.spinner("Manually calling the LLM..."):
                    formatted_prompt = qa_prompt.format(context=context_to_test, question=question_to_test)
                    llm = st.session_state.rag_chain.combine_docs_chain.llm_chain.llm
                    result = llm.invoke(formatted_prompt)
                    st.info("LLM Output:"); st.write(result.content)