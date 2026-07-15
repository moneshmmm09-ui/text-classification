import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="Text Classification App",
    page_icon="🤖",
    layout="centered"
)

# Load Hugging Face Model
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# Title
st.title("🤖 Hugging Face Text Classification")
st.write("Enter text below to classify it as Positive or Negative.")

# Text Input
text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Example: I love this product!"
)

# Button
if st.button("Classify"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            result = classifier(text)

            label = result[0]["label"]
            score = result[0]["score"]

            st.subheader("Prediction")

            if label == "POSITIVE":
                st.success("😊 Positive")
            else:
                st.error("😞 Negative")

            st.write(f"**Confidence:** {score * 100:.2f}%")
            st.progress(float(score))

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
Model:
- distilbert-base-uncased-finetuned-sst-2-english

Task:
- Text Classification

Library:
- Hugging Face Transformers

Framework:
- Streamlit
""")
