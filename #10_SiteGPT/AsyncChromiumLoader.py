from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

st.set_page_config (
    page_title = "SiteGPT",
    page_icon="##",
)

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    # SiteGPT
    """
)

with st.sidebar:
    url = st.text_input(
        "URL을 입력하세요.",
        placeholder="http://example.com"
    )

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)