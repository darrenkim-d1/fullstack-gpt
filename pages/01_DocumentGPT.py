from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os

st.set_page_config(
    page_title="DocumentGPT",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    api_key=st.sidebar.text_input("openai key")
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()

    # 1. 저장할 디렉토리 경로 설정
    cache_dir = "./.cache/files" 
    
    # 2. 해당 디렉토리가 없으면 생성 (exist_ok=True로 중복 생성 방지)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    file_path = os.path.join(cache_dir, file.name)
    
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True): 
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
           Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
           
           Context: {context}
           """,
),
("human", "{question}"),
]
)


st.title("DocumentGPT")

st.markdown(
"""
Welcome!
"""
)

with st.sidebar:
    st.write("---")
    st.link_button("github commit", "https://github.com/darrenkim-d1/fullstack-gpt/commit/b22e49ce9ecca809ea2febea1f87120ab6ba8ca4")
    st.write("---")

    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        chain = (
            {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []