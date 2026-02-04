import streamlit as st

st.title("Hello world!")

st.write([1,2,3])

st.markdown(
    """
    #1. Title
    ---
    | 개요

    ---
    ```
    #include <stdio.h>
    printf("안녕하세요")
    ```
    ---
    """
)

st.selectbox("모델을 선택하세요.", ("GPT-3", "GPT-4"))