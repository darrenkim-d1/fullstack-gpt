from langchain.document_loaders import SitemapLoader, text
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

llm = ChatOpenAI(
    temperature=0.1,
    api_key=st.sidebar.text_input("openai key")
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    오직 아래의 **컨텍스트(Context)**만을 사용하여 사용자의 질문에 답하세요. 답을 알 수 없다면 단순히 모르겠다고 말하고, 내용을 지어내지 마세요.

    그다음, 답변에 대해 0점에서 5점 사이의 점수를 매기세요.

    답변이 사용자의 질문에 제대로 답했다면 높은 점수를, 그렇지 않다면 낮은 점수를 주어야 합니다.

    답변 점수가 0점일지라도 반드시 점수를 포함하세요.

    컨텍스트: {context}

    예시:

    질문: 달은 얼마나 멀리 있나요?
    답변: 달은 384,400km 떨어져 있습니다.
    점수: 5

    질문: 태양은 얼마나 멀리 있나요?
    답변: 모르겠습니다.
    점수: 0

    당신 차례입니다!

    질문: {question}
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    # st.write(answers)

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            사용자의 질문에 답할 때, 오직 아래에 제공된 기존 답변들만을 사용하십시오.

            여러 답변 중 점수가 가장 높고(가장 도움이 되는) 최신 상태인 답변을 우선적으로 선택하여 답변하십시오.

            반드시 출처를 인용하되, 답변의 출처는 내용을 수정하지 말고 원문 그대로 반환하십시오.

            답변 데이터: {answers}
            """
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


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
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            f"https://developers.cloudflare.com/ai-gateway/",
            f"https://developers.cloudflare.com/vectorize/",
            f"https://developers.cloudflare.com/workers-ai/",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)

    if len(docs) > 100:
        docs = docs[:100] # 상위 100개 청크만 사용 (약 10만 토큰 내외)

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever(search_kwargs={"k": 2})


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    st.write("---")
    st.link_button("github commit", "https://github.com/darrenkim-d1/fullstack-gpt/commit/4167021e36447f9636064465e2ab758f7b85e06c")
    st.write("---")
    url = st.text_input(
        "Write down a URL",
        value="https://developers.cloudflare.com/sitemap-0.xml",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)

        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))