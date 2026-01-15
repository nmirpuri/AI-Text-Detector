import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Detector")

st.title("AI Text Detector")

# Load model (this may take a few seconds)
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="roberta-base-openai-detector")

detector = load_model()

# User input
text = st.text_area("Paste text here:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            result = detector(text)[0]
            st.success(f"Prediction: {result['label']}")
            st.info(f"Confidence: {round(result['score']*100, 2)}%")
