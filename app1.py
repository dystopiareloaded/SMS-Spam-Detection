import streamlit as st
import pickle
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# ---------------------- Download NLTK stopwords only ----------------------
def download_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

download_nltk_resources()

# ---------------------- Load Model and Vectorizer ------------------------
try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

ps = PorterStemmer()

# ---------------------- Text Transformation Function ---------------------
def transform_text(text):
    text = text.lower()

    # Use RegexpTokenizer instead of word_tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

    return " ".join(stemmed_tokens)

# ---------------------- Streamlit App UI ---------------------------------
st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="centered")

# Custom CSS and creator credit
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextArea > label {
        font-size: 1.2rem;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 0.85rem;
    }
    </style>
    <div class="footer">
        📌 Created by <strong>Kaustav Roy Chowdhury</strong>
    </div>
""", unsafe_allow_html=True)

st.title("📩 SMS Spam Detection")
st.markdown("Enter a message below and find out if it's **spam** or **not**! 💡")

input_sms = st.text_area("Enter your message:", height=150)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0][result]

        # Output
        if result == 1:
            st.error(f"🚨 This is **SPAM**! ({prob*100:.2f}% confidence)")
        else:
            st.success(f"✅ This is **Not Spam**. ({prob*100:.2f}% confidence)")
