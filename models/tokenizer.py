import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sqlalchemy import create_engine


def tokenize(text):
    '''
    Tokenizes text using lemmatization.
    
    Args:
    - text: A text string.
    
    Returns:
    - clean_tokens: Cleaned and tokenized text.
    '''
    # Create url placeholders
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Remove numeric
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    
    words = tokenizer.tokenize(text.lower())
    # Remove stopwords
    tokens = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    # Lemmatize tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()        
        clean_tokens.append(clean_tok)

    return clean_tokens