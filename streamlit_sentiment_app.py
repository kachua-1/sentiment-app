# streamlit_sentiment_app.py
# -------------------------------------------------------------
# Streamlit app to fetch Google Play reviews and perform sentiment analysis
# using a trained Keras model hosted on Hugging Face.

import streamlit as st
from google_play_scraper import reviews
import pandas as pd
import numpy as np
import os
import re
import json
from textblob import TextBlob
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from huggingface_hub import hf_hub_download

# Download NLTK data quietly
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ====== CONFIG ======
HF_REPO_ID = "soapmac123/sentiment-model"  # 👈 change if your HF repo name is different
HF_MODEL_FILENAME = "sentiment_model.h5"
HF_LABELMAP_FILENAME = "label_map.json"  # optional, will be ignored if missing

# ====== Streamlit UI ======
st.set_page_config(layout='wide', page_title='App Reviews Sentiment Explorer')
st.title('📱 App Reviews — Sentiment Explorer')
st.write('Enter a Google Play app id (for example `com.whatsapp`) and run sentiment analysis using your trained ML model.')

col1, col2 = st.columns([2,1])

with col1:
    app_id = st.text_input('Google Play App ID', value='com.whatsapp')
    n_reviews = st.number_input('Number of reviews to fetch', min_value=10, max_value=2000, value=200, step=10)
    lang = st.selectbox('Review language', options=['en','all'], index=0)
    country = st.text_input('Country code (two letters)', value='us')

with col2:
    st.markdown('This app automatically downloads your model from Hugging Face.')
    st.markdown('✅ Make sure you’ve uploaded `sentiment_model.h5` to your HF model repo.')

run = st.button('Run sentiment analysis')

# ====== Helper functions ======
def fetch_google_play_reviews(app_id: str, lang='en', country='us', count=200):
    """Fetch reviews using google_play_scraper"""
    st.info(f"Fetching up to {count} reviews for {app_id}...")
    all_reviews, continuation_token = [], None
    fetched = 0
    while fetched < count:
        batch = min(200, count - fetched)
        result, continuation_token = reviews(app_id, lang=lang, country=country, count=batch, continuation_token=continuation_token)
        if not result:
            break
        all_reviews.extend(result)
        fetched += len(result)
        if continuation_token is None:
            break
    df = pd.DataFrame(all_reviews)
    if df.empty:
        st.warning("No reviews fetched. Check the app id and try again.")
    else:
        df = df.rename(columns={'content': 'review'})[['userName', 'review', 'score', 'at']]
    return df

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\\S+|www\\S+', '', text)
    text = re.sub(r'[^a-z0-9\\s]', ' ', text)
    return re.sub(r'\\s+', ' ', text).strip()

def preprocess_reviews(series: pd.Series) -> pd.Series:
    lemma = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    def f(x):
        x = clean_text(str(x))
        tokens = nltk.word_tokenize(x)
        tokens = [lemma.lemmatize(t) for t in tokens if t.isalpha() and t not in stops]
        return ' '.join(tokens)
    return series.map(f)

def textblob_sentiment(text: str) -> str:
    p = TextBlob(text).sentiment.polarity
    if p > 0.1: return 'positive'
    elif p < -0.1: return 'negative'
    else: return 'neutral'

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_huggingface_model():
    try:
        st.info('📥 Downloading model from Hugging Face...')
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME, repo_type="model")
        labelmap_path = None
        try:
            labelmap_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_LABELMAP_FILENAME, repo_type="model")
        except Exception:
            labelmap_path = None
        model = load_model(model_path)
        label_map = {}
        if labelmap_path and os.path.exists(labelmap_path):
            with open(labelmap_path, 'r') as f:
                label_map = json.load(f)
        st.success('✅ Model loaded successfully from Hugging Face!')
        return model, label_map
    except Exception as e:
        st.warning(f"⚠️ Could not load model from Hugging Face. Falling back to TextBlob. Error: {e}")
        return None, None

if run and app_id:
    df = fetch_google_play_reviews(app_id, lang=(None if lang=='all' else lang), country=country, count=int(n_reviews))
    if df.empty:
        st.stop()

    df['clean_review'] = preprocess_reviews(df['review'])

    model, label_map = load_huggingface_model()

    if model is not None:
        st.info('Running predictions using your trained model...')
        embedder = load_sentence_model()
        embeddings = embedder.encode(df['clean_review'].tolist(), show_progress_bar=True)
        preds = model.predict(np.array(embeddings))

        if preds.ndim == 1 or preds.shape[1] == 1:
            probs = preds.squeeze()
            df['predicted_sentiment'] = ['positive' if p >= 0.5 else 'negative' for p in probs]
        else:
            labels_idx = preds.argmax(axis=1)
            df['predicted_sentiment'] = [label_map.get(str(int(i)), f'class_{i}') for i in labels_idx]
    else:
        st.info('Falling back to TextBlob sentiment analysis...')
        df['predicted_sentiment'] = df['review'].map(textblob_sentiment)

    # ====== Visualization ======
    st.subheader('📊 Results Summary')
    counts = df['predicted_sentiment'].value_counts().reindex(['positive', 'neutral', 'negative']).fillna(0).astype(int)
    st.bar_chart(counts)

    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Sample reviews
    st.subheader('🗒 Sample Reviews')
    for sentiment in ['positive', 'neutral', 'negative']:
        st.write(f"### {sentiment.capitalize()} Reviews")
        st.dataframe(df[df['predicted_sentiment']==sentiment][['userName','review','score','at']].head(10))

    csv = df.to_csv(index=False)
    st.download_button('💾 Download Results CSV', data=csv, file_name=f'{app_id}_sentiment_results.csv', mime='text/csv')

else:
    st.info('Enter an App ID and click **Run sentiment analysis** to start.')

st.markdown('---')
st.caption('This app automatically downloads your trained model from Hugging Face and uses it to analyze Google Play reviews. Make sure `HF_REPO_ID` matches your model repo, and Streamlit has access to your Hugging Face token if the repo is private.')
