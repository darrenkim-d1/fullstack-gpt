from langchain.document_loaders import SitemapLoader
import streamlit as st

@st.cache_data(show_spinner="로딩중..")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 5
    docs = loader.load()
    return docs

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
    url = st.text_input(
        "URL을 입력하세요."
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Sitemap URL을 입력하세요.")
    else:
        docs = load_website(url)
        st.write(docs)