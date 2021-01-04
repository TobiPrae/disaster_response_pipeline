import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from models.tokenizer import tokenize

def main():
    
    # Get data
    print("...load data")
    engine = create_engine('sqlite:///../disaster_response_pipeline/data/DisasterResponse.db')
    df = pd.read_sql_table('cleaned_messages', engine)
    
    # Tokenization
    print("...get message lengths")
    df["tokenized"] = df["message"].apply(tokenize)
    df["message_lengths"] = df["tokenized"].apply(len)
    df_message_lengths = df.drop(columns=["tokenized"])
    df_message_lengths.to_sql('df_message_lengths', engine, index=False, if_exists="replace")
    
    print("...get top categories")
    # Top 10 categories
    df_categories = df.iloc[:, 4:-2]
    category_count_df = pd.DataFrame(df_categories.apply(pd.Series.value_counts).iloc[1, :]).sort_values(by=1, ascending=False)
    category_count_df.to_sql('category_count_df', engine, index=False, if_exists="replace")
    
    print("...get categories per row")
    # N categories histogram
    n_categories = [sum(df_categories.iloc[i, :].tolist()) for i in range(df_categories.shape[0])]
    n_categories_df = pd.DataFrame(n_categories, columns=["count"])
    n_categories_df.to_sql('n_categories_df', engine, index=False, if_exists="replace")
    
    print("...get word count for aid")
    # Word count aid
    df_aid = df.loc[df["aid_related"] == 1]
    words_aid = df_aid["tokenized"].sum(axis=0)
    values_aid, counts_aid = np.unique(words_aid, return_counts=True)
    df_word_counts_aid = pd.DataFrame(list(zip(values_aid, counts_aid)), columns=["word", "count"]).sort_values(by="count", ascending=False)
    df_word_counts_aid.to_sql('df_word_counts_aid', engine, index=False, if_exists="replace")
    
    print("...get word count for weather")
    # Word count weather
    df_weather = df.loc[df["weather_related"] == 1]
    words_weather = df_weather["tokenized"].sum(axis=0)
    values_weather, counts_weather = np.unique(words_weather, return_counts=True)
    df_word_counts_weather = pd.DataFrame(list(zip(values_weather, counts_weather)), columns=["word", "count"]).sort_values(by="count", ascending=False)
    df_word_counts_weather.to_sql('df_word_counts_weather', engine, index=False, if_exists="replace")
    
if __name__ == '__main__':
    main()