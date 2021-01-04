import sys
import re
import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sqlalchemy import create_engine

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs
from plotly.subplots import make_subplots

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib



app = Flask(__name__)

# Custom transformers

class WordCounter(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_word_count = pd.Series(X).apply(lambda x : len(x.split()))
        return pd.DataFrame(X_word_count)

class CharCounter(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_char_count = pd.Series(X).apply(lambda x : len(x.replace(" ","")))
        return pd.DataFrame(X_char_count)

class UniqueWordCounter(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_unique_count = pd.Series(X).apply(lambda x: len(set(word for word in x.split())))
        return pd.DataFrame(X_unique_count)

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_text_length = pd.Series(X).apply(lambda x : len(x))
        return pd.DataFrame(X_text_length)

class CapitalCounter(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_capital_count = pd.Series(X).apply(lambda x: sum(1 for char in x if char.isupper()))
        return pd.DataFrame(X_capital_count)
      
class CapLengthRatio(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_capital_count = pd.Series(X).apply(lambda x: sum(1 for char in x if char.isupper()))
        X_text_length = pd.Series(X).apply(lambda x : len(x))
        X_cap_len_ratio = X_capital_count/X_text_length
        return pd.DataFrame(X_cap_len_ratio)
    
class SpecificWordCounter(BaseEstimator, TransformerMixin):
    
    def __init__(self, word):
        self.word = word.lower()
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_n = pd.Series(X).apply(lambda x: x.lower().count(self.word))
        return pd.DataFrame(X_n)
 
class SymbolCounter(BaseEstimator, TransformerMixin):
    
    def __init__(self, symbols):
        self.symbols = symbols
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_n = pd.Series(X).apply(lambda x: sum(x.count(s) for s in self.symbols))
        return pd.DataFrame(X_n)

def tukey_rule(df, col):
    '''
    Applies Tukey rule and removes outliers from data.
    
    Args:
    - df: Dataframe
    - col: Column
    
    Returns:
    - df_new: Cleaned df
    
    '''
    X = df[col]

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR
    
    df_new = df.loc[(df[col] < max_value) & (df[col] > min_value)]
    
    return df_new

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

# load and prepare data
engine = create_engine('sqlite:///../disaster_response_pipeline/data/DisasterResponse.db')
df = pd.read_sql_table('cleaned_messages', engine)

# Number of categories per message
n_categories_df = pd.read_sql_table('n_categories_df', engine)
n_categories = n_categories_df["count"].tolist()

# Categories by occurence
df_categories = df.iloc[:, 3:-1]
category_count_df = pd.DataFrame(df_categories.apply(pd.Series.value_counts).iloc[1, :]).sort_values(by=1, ascending=False)

# Word lengths
df_plot = pd.read_sql_table('df_message_lengths', engine)
df_without_outliers = tukey_rule(df_plot, "message_lengths")

# Word counts
df_word_counts_aid = pd.read_sql_table('df_word_counts_aid', engine)
df_word_counts_weather = pd.read_sql_table('df_word_counts_weather', engine) 

# Load model
model = joblib.load("../disaster_response_pipeline/models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    graphs = []
    
    # figure 1
    fig1 = plotly.graph_objs.Figure([plotly.graph_objs.Bar(x=category_count_df[1].tolist()[:10],
                        y=category_count_df.index.tolist()[:10],
                        orientation='h',
                        marker_color='indianred')])
    fig1.update_layout(
        yaxis_title='Category',
        xaxis_title='Count',
        title='Top 10 categories sorted by occurence',
        hovermode="y",
        template="plotly_white"
    )

    fig1['layout']['yaxis']['autorange'] = "reversed"    
    
    graphs.append(fig1)
    
    # figure 2
    fig2 = plotly.graph_objs.Figure(data=[plotly.graph_objs.Histogram(x=n_categories,
                                   histnorm='probability',
                                   marker_color='lightsalmon',
                                   xbins=dict(
                                       size=1
                                   ),
                                  )
                     ])

    fig2.update_layout(
        yaxis_title='Probability',
        xaxis_title='Category count',
        title='Number of categories per message (Probability)',
        hovermode="x",
        template="plotly_white"
    )
    
    graphs.append(fig2)
    
    # figure 3
    fig3 = plotly.graph_objs.Figure()

    for col in category_count_df.index.tolist()[:10]:
        df_filter = df_without_outliers.loc[df_without_outliers[col] == 1]

        fig3.add_trace(plotly.graph_objs.Box(y=df_filter["message_lengths"],
                             name=F"{col}",
                             boxpoints=False,
                             boxmean=True))

    fig3.update_layout(
        yaxis_title='Word count',
        xaxis_title='Category',
        title='Word count per category and message without outliers (Top 10 most common categories)',
        hovermode="x",
        template="plotly_white",
)

    graphs.append(fig3)
    
    
    
    # figure 4
    fig4 = plotly.graph_objs.Figure([plotly.graph_objs.Bar(x=df_word_counts_aid["word"].head(10),
                        y=df_word_counts_aid["count"].head(10),
                        marker_color='indianred')])
    
    fig4.update_layout(
        yaxis_title='Count',
        xaxis_title='Word',
        title='Top 10 words for category "aid_related"',
        hovermode="x",
        template="plotly_white"
    )
    
    graphs.append(fig4)

    # figure 5
    fig5 = plotly.graph_objs.Figure([plotly.graph_objs.Bar(x=df_word_counts_weather["word"].head(10),
                        y=df_word_counts_weather["count"].head(10),
                        marker_color='lightsalmon')])
    
    fig5.update_layout(
        yaxis_title='Count',
        xaxis_title='Word',
        title='Top 10 words for category "weather_related"',
        hovermode="x",
        template="plotly_white"
    )

    graphs.append(fig5)
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()