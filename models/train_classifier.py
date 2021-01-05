import sys
import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sqlalchemy import create_engine

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from models.tokenizer import tokenize

import pickle
from functools import partial

# Custom transformers

class WordCounter(BaseEstimator, TransformerMixin):
    '''
    Counts number of words in given text.
    '''
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_word_count = pd.Series(X).apply(lambda x : len(x.split()))
        return pd.DataFrame(X_word_count)

    
class CharCounter(BaseEstimator, TransformerMixin):
    '''
    Counts number of chars in given text.
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_char_count = pd.Series(X).apply(lambda x : len(x.replace(" ","")))
        return pd.DataFrame(X_char_count)


class UniqueWordCounter(BaseEstimator, TransformerMixin):
    '''
    Counts number of unique words in given text.
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_unique_count = pd.Series(X).apply(lambda x: len(set(word for word in x.split())))
        return pd.DataFrame(X_unique_count)

    
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    Extracts length for given text.
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_text_length = pd.Series(X).apply(lambda x : len(x))
        return pd.DataFrame(X_text_length)


class CapitalCounter(BaseEstimator, TransformerMixin):
    '''
    Counts number of capitals in given text.
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_capital_count = pd.Series(X).apply(lambda x: sum(1 for char in x if char.isupper()))
        return pd.DataFrame(X_capital_count)
    
    
class CapLengthRatio(BaseEstimator, TransformerMixin):
    '''
    Compares text length to number of capiatls in given text.
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_capital_count = pd.Series(X).apply(lambda x: sum(1 for char in x if char.isupper()))
        X_text_length = pd.Series(X).apply(lambda x : len(x))
        X_cap_len_ratio = X_capital_count/X_text_length
        return pd.DataFrame(X_cap_len_ratio)
    
    
class SpecificWordCounter(BaseEstimator, TransformerMixin):
    '''
    Counts the occurence of specific words in given text.
    '''
    def __init__(self, word):
        self.word = word.lower()
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_n = pd.Series(X).apply(lambda x: x.lower().count(self.word))
        return pd.DataFrame(X_n)

   
class SymbolCounter(BaseEstimator, TransformerMixin):
    '''
    Counts the occurence of given symbols in given text.
    '''
    def __init__(self, symbols):
        self.symbols = symbols
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_n = pd.Series(X).apply(lambda x: sum(x.count(s) for s in self.symbols))
        return pd.DataFrame(X_n)


# Functions
    
def load_data(database_filepath):
    '''
    Loads data from database and extracts X and Y.
    
    Args:
    - data_base_filepath: Filepath to sqlite-Database
    
    Returns:
    - X: Preprocessed messages
    - Y: Labels
    - category_names: column names of label columns
    '''
    engine = create_engine(F'sqlite:///{database_filepath}')
    df = pd.read_sql_table('cleaned_messages', engine)
    X = df["message"].values
    Y = df[df.iloc[:, 4:].columns].values
    category_names =df.iloc[:, 4:].columns.tolist()
    
    return X, Y, category_names



def build_model():
    '''
    Builds pipeline and grid search object.
    
    Args: None
    
    Returns
    - Grid search object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=partial(tokenize))),
                ('tfidf', TfidfTransformer())
            ])),

            # basic text analysis
            ('text_length', TextLengthExtractor()),
            ('wourd_count', WordCounter()),
            ('char_count', CharCounter()),
            ('capital_count', CapitalCounter()),
            ('cap_len_ratio', CapLengthRatio()),
            ('unique_word_counter', UniqueWordCounter()),

            # count symbols
            ('exclamationmark_counter', SymbolCounter('!')),
            ('questionmark_counter', SymbolCounter('?')),
            ('hashtag_counter', SymbolCounter('#')),
            ('symbol_counter', SymbolCounter('*&$%')),
            ('punctuation_counter', SymbolCounter('.,;:')),
            ('number_counter', SymbolCounter('0123456789')),

            # count words
            ('sos_counter', SpecificWordCounter('sos')),
            ('please_counter', SpecificWordCounter('please')),
            ('help_counter', SpecificWordCounter('help')),
            ('emergency_counter', SpecificWordCounter('emergency')),

        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=25)))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)],
    }

    cv = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=10)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predicts labels, evaluates model and prints results.
    
    Args:
    - model: Trained model
    - X_test: Test set X
    - Y_test: Test set Y
    - category_names: Column names of label columns
    
    Returns: Nothing
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    Y_test = pd.DataFrame(Y_test, columns=category_names)
    
    f1_scores = []
    
    for col in category_names:
        report = classification_report(Y_test[col], Y_pred[col], labels=[1, 0])
        f1 = f1_score(list(Y_test[col].astype(int)), list(Y_pred[col].astype(int)), average="weighted")
        f1_scores.append(f1)
        print(report)
        print("__________")
    avg_f1 = sum(f1_scores)/len(f1_scores)
    print(F"Average model f1 score: {avg_f1}")
        
        
        
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
