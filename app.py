import streamlit as st
st.set_page_config(page_title="🎬 Sentiment Analyzer", layout="centered")

import joblib
import re
import string

# Add Background and Style to the App
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            border: none;
            font-weight: bold;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

#  Load Assets 
@st.cache_resource
def load_assets():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_assets()

#  Clean Text 
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r"\d+", '', text)
    return text.strip()

#  UI Layout 
st.markdown("<h1 style='text-align: center;'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("Give a movie review below, and the model will predict its sentiment.")

# Input
user_input = st.text_area("✍️ Write your review here:", height=150)

# Prediction
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "🟢 Positive" if prediction == 1 else "🔴 Negative"
        st.success(f"### Predicted Sentiment: {sentiment}")

# Expanders
with st.expander("📊 How it Works"):
    st.markdown("""
        - **Model**: Logistic Regression  
        - **Training Data**: IMDB movie reviews  
        - **Vectorization**: TF-IDF  
        - **Labels**: `1` = Positive, `0` = Negative  
    """)

with st.expander("⚠️ Possible Errors / Limitations"):
    st.markdown("""
        - May misread sarcasm or jokes  
        - Needs enough context to understand intent  
        - Short reviews might be less accurate  
    """)

# Footer
st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)
st.markdown("<center><sub>Built with ❤️ using Streamlit</sub></center>", unsafe_allow_html=True)
