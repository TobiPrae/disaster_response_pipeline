# disaster_response

**Introduction**

Hi, welcome to this project. I created this repository as a part of the data science nanodegree on udacity.

**Description**

The goal of the project is to classify messages collected in the context of disasters. These messages should then be assigned to a category, where each message can have multiple labels. For this purpose, ETL and ML pipelines should be combined. I have tested different classifiers from scikit-learn based on an average weighted F1 score compared with each other. I chose the AdaBoostClassifier, which is an ensemble algorithm that uses decision trees as a base estimator.

**Instructions**
1. To get started clone this repository with 
`git clone https://github.com/TobiPrae/disaster_response_pipeline.git`

2. Installations.
    - `pip install plotly==4.13.0`

3. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run data preparation for visualization and store results in db
        `python data/prepare_plots.py`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Run the following command in the app's root directory to run your web app.
    `python app/run.py`

5. Go to http://0.0.0.0:3001/