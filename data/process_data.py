import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads categories and messages and merges them by id.
    
    Args:
    - messages_filepath: Filepath for message df
    - categories_filepath: Filepath for categories df
    
    Returns:
    - df: Merged dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    ids = categories["id"]
    categories = categories["categories"].str.split(";", expand=True)
    
    row = categories.iloc[0, :].tolist()

    category_colnames = [element.split("-")[0] for element in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
       
        # Fix related column
        categories[column] = categories[column].replace(2, 1)
    
    categories["id"] = ids

    df = pd.merge(messages, categories, left_on='id', right_on='id', how='left')
    
    return df
    
    
def clean_data(df):
    '''
    Cleans data from duplicates and from unnecessary columns:
    
    Args: 
    - df: Merged dataframe
    
    Returns:
    - Cleaned df
    '''
    # Finds and removes duplicates
    df["duplicate"] = df.duplicated(subset="id", keep='first')
    df = df.loc[df["duplicate"] == False]
    df = df.drop(columns="duplicate")

    # Drops column "original" as it is not relevant for text classification
    df = df.drop(columns="original")
    
    # Drops column "child_alone" as it contains only "0" as value
    df = df.drop(columns="child_alone")
    
    return df




def save_data(df, database_filename):
    '''
    Saves df to sqlite db
    
    Args:
    - df: Cleaned df
    - database_filename: Filename of db
    
    Returns:
    - Nothing
    '''
    engine = create_engine(F"sqlite:///{database_filename}") 
    df.to_sql('cleaned_messages', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()