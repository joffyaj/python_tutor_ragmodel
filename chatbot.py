import os
import torch
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
st.set_page_config(
    page_title="AI Python Tutor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stChatMessage {
            font-size: 15px;
            line-height: 1.6;
        }
        .tutor-box {
            background-color: #f6f8fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4f46e5;
            color: #111827;   /* FIX: visible text */
        }
        .tutor-box p,
        .tutor-box li,
        .tutor-box span {
            color: #111827;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# -------------------------------------------------
torch.set_num_threads(4)

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
        st.experimental_rerun()

    st.markdown(
        """
        **About This Tutor**

        • Python & Data Structures only  
        • RAG-based (no hallucinations)  
        • Challenges shallow answers  
        • CPU-only demo deployment  
        """
    )

# -------------------------------------------------
st.markdown("## 🎓 AI Tutor for Python & Data Structures")
st.markdown(
    "Explain concepts in your own words — expect the tutor to challenge you."
)
st.markdown("---")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

@st.cache_resource
def load_vectorstore():
    if not os.path.exists("vectorstore"):
        st.info("🔄 Building knowledge base for the first time...")

        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        loader = PyPDFLoader("data/data.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("vectorstore")

        return vectorstore

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 2})

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    template="""
You are an AI Tutor teaching Python and Data Structures.

Teaching style: {tutor_mode}
Difficulty level: {difficulty}

Rules:
- Do NOT give shallow explanations.
- If the student's input is vague, push them to explain why or how.
- If the answer is correct but superficial, challenge it.
- If the answer is not found in the context, say "I don't know."

Context:
{context}

Student Input:
{question}

Tutor Response:
""",
    input_variables=["context", "question", "tutor_mode", "difficulty"]
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "tutor_mode": lambda _: tutor_mode,
        "difficulty": lambda _: difficulty
    }
    | prompt
    | llm
    | StrOutputParser()
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


user_input = st.chat_input("Ask a question or explain a concept...")

if user_input:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Tutor response
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