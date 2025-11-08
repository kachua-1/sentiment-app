import streamlit as st
import traceback

# Safe startup
try:
    st.set_page_config(layout="wide", page_title="App Reviews Sentiment Explorer")
except Exception as e:
    st.error("Startup failed before UI rendered!")
    st.text(traceback.format_exc())
    st.stop()

st.title("📱 App Reviews — Sentiment Explorer")
st.caption("Runs your trained model from Hugging Face on Google Play reviews")

# User input section
app_id = st.text_input("Google Play App ID", "com.whatsapp")
n_reviews = st.number_input("Number of reviews to fetch", min_value=10, max_value=1000, value=100, step=50)
run = st.button("Run sentiment analysis")

# --- Only import heavy libraries after button click ---
if run and app_id:
    with st.spinner("Loading dependencies... please wait"):
        import pandas as pd, numpy as np, re, json, time
        from google_play_scraper import reviews
        from sentence_transformers import SentenceTransformer
        from tensorflow.keras.models import load_model
        from huggingface_hub import hf_hub_download
        from textblob import TextBlob
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

    st.success("✅ Libraries loaded")

    # Load model safely
    @st.cache_resource
    def load_huggingface_model():
        try:
            st.info("Downloading model from Hugging Face...")
            path = hf_hub_download(repo_id="soapmac123/sentiment",
                                   filename="sentiment_model.h5", repo_type="model")
            model = load_model(path)
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.text(traceback.format_exc())
            return None

    model = load_huggingface_model()
    if model is None:
        st.stop()

    # Fetch reviews with timeout + progress
    def fetch_reviews(app_id, count):
        from google_play_scraper import reviews
        all_reviews, token = [], None
        progress = st.progress(0)
        start = time.time()
        while len(all_reviews) < count:
            batch = min(200, count - len(all_reviews))
            try:
                data, token = reviews(app_id, count=batch, continuation_token=token)
                all_reviews.extend(data)
                progress.progress(min(1.0, len(all_reviews) / count))
                if token is None or time.time() - start > 90:
                    break
            except Exception as e:
                st.warning(f"Stopped early: {e}")
                break
        progress.empty()
        return pd.DataFrame(all_reviews)

    st.info("Fetching reviews...")
    df = fetch_reviews(app_id, n_reviews)
    if df.empty:
        st.error("No reviews fetched.")
        st.stop()

    # Preprocess
    lemma = WordNetLemmatizer()
    stops = set(stopwords.words("english"))
    def clean(text):
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        return " ".join([lemma.lemmatize(w) for w in nltk.word_tokenize(text) if w.isalpha() and w not in stops])
    df["clean"] = df["content"].map(clean)

    # Embeddings and prediction
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    st.info("Generating embeddings...")
    emb = embedder.encode(df["clean"].tolist(), show_progress_bar=False, batch_size=16)
    st.info("Running predictions...")
    preds = model.predict(emb)
    df["sentiment"] = ["positive" if p >= 0.5 else "negative" for p in preds.squeeze()]

    st.bar_chart(df["sentiment"].value_counts())
    st.dataframe(df[["userName", "content", "sentiment"]])
    st.success("✅ Done!")
else:
    st.info("Enter an App ID and click 'Run sentiment analysis' to start.")

