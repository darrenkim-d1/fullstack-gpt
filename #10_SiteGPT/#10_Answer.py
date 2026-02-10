from langchain.document_loaders import SitemapLoader, AsyncChromiumLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

llm = ChatOpenAI(
    temperature=0.1
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answer(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    answers = []
    for doc in docs:
        result = answers_chain.invoke(
            {"question": question, "context": doc.page_content}
        )
        answers.append(result.content)
    st.write(answers)

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

@st.cache_data(show_spinner="로딩중")
def load_website(url):
    spitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    loader = AsyncChromiumLoader([url])
    docs = loader.load_and_split(text_splitter=spitter)
    
    Html2TextTransformer().transform_documents(docs)

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config (
    page_title = "SiteGPT",
    page_icon="##",
)

st.markdown(
    """
    # Site GPT
    """
)

with st.sidebar:
    url2 = st.text_input(
        label="llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?",
        value="https://developers.cloudflare.com/workers-ai/"
    )

    url = st.text_input(
        label="Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?",
        value="https://developers.cloudflare.com/ai-gateway/"
    )

    url3 = st.text_input(
        label="벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?",
        value="https://developers.cloudflare.com/vectorize/"
    )

if url2:
    retriever2 = load_website(url2)

    chain2 = {
        "docs": retriever2,
        "question": RunnablePassthrough(),
    } | RunnableLambda(get_answer)

    chain2.invoke("llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?")

if url:
    retriever = load_website(url)

    chain = {
        "docs": retriever,
        "question": RunnablePassthrough(),
    } | RunnableLambda(get_answer)

    chain.invoke("Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?")



if url3:
    retriever3 = load_website(url3)

    chain3 = {
        "docs": retriever3,
        "question": RunnablePassthrough(),
    } | RunnableLambda(get_answer)

    chain3.invoke("벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?")    