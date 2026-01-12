import os
import torch
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Python Tutor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Global Styling (Text Visibility FIXED)
# -------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .stChatMessage { font-size: 15px; line-height: 1.6; }
        .tutor-box {
            background-color: #f6f8fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4f46e5;
            color: #111827;
        }
        .tutor-box p, .tutor-box li, .tutor-box span {
            color: #111827;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Imports
# -------------------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------------------------
# Performance Tuning
# -------------------------------------------------
torch.set_num_threads(4)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.title("🎓 AI Tutor Settings")

    tutor_mode = st.selectbox(
        "Teaching Style",
        ["Strict Professor", "Guided Mentor", "Beginner Friendly"]
    )

    difficulty = st.selectbox(
        "Difficulty Level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    st.markdown("---")

    if st.button("🧹 Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
        **About This Tutor**

        • Python & Data Structures  
        • RAG-based (context grounded)  
        • Explains questions  
        • Challenges shallow answers  
        • Free CPU deployment  
        """
    )

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("## 🎓 AI Tutor for Python & Data Structures")
st.markdown("Ask questions or explain concepts — the tutor will respond accordingly.")
st.markdown("---")

# -------------------------------------------------
# Intent Detection (CRITICAL FIX)
# -------------------------------------------------
def detect_intent(user_input: str) -> str:
    question_keywords = (
        "what", "why", "how", "explain", "define", "tell", "difference", "describe"
    )
    if user_input.lower().strip().startswith(question_keywords):
        return "question"
    return "answer"

# -------------------------------------------------
# Load Embeddings
# -------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -------------------------------------------------
# Load or Build Vectorstore (DEPLOYMENT FIX)
# -------------------------------------------------
@st.cache_resource
def load_vectorstore():
    if not os.path.exists("vectorstore"):
        st.info("🔄 Building knowledge base for the first time...")

        loader = PyPDFLoader("data/data.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local("vectorstore")
        return vs

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 2})

# -------------------------------------------------
# Load LLM
# -------------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# -------------------------------------------------
# Helper
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------
# Prompt (Intent-Aware Tutor)
# -------------------------------------------------
prompt = PromptTemplate(
    template="""
You are an AI Tutor teaching Python and Data Structures.

Teaching style: {tutor_mode}
Difficulty level: {difficulty}
Student intent: {intent}

Rules:
- If intent is QUESTION:
  - Explain clearly and step by step.
  - Do NOT challenge the student.
- If intent is ANSWER:
  - If correct but shallow, challenge it.
  - If incomplete, ask why or how.
- If the information is not found in context, say "I don't know".

Context:
{context}

Student Input:
{question}

Tutor Response:
""",
    input_variables=["context", "question", "tutor_mode", "difficulty", "intent"]
)

# -------------------------------------------------
# RAG Chain
# -------------------------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "tutor_mode": lambda _: tutor_mode,
        "difficulty": lambda _: difficulty,
        "intent": lambda x: detect_intent(x)
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------------------------
# Chat Memory
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
user_input = st.chat_input("Ask a question or explain a concept...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    response = rag_chain.invoke(user_input)

    tutor_response = f"""
    <div class="tutor-box">
    {response}
    </div>
    """

    st.session_state.messages.append(
        {"role": "assistant", "content": tutor_response}
    )
    with st.chat_message("assistant"):
        st.markdown(tutor_response, unsafe_allow_html=True)
