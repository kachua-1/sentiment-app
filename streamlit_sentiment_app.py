# Download necessary NLTK data (safe and version-agnostic)
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# for newer versions of NLTK
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
