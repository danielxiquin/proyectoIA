import re 
import string 
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        # Eliminar placeholders tipo {{Order Number}}, {{Customer Name}}, etc.
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        return text
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]
    
    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

