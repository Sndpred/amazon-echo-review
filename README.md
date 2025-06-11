**Amazon Echo Reviews Analysis**

**Project Overview**
This project involves analyzing Amazon Echo reviews to gain insights into customer sentiment and identify key themes, particularly distinguishing between positive and negative feedback. The analysis utilizes natural language processing (NLP) techniques and machine learning models to classify reviews and extract meaningful information.

**Objectives**

    To perform exploratory data analysis (EDA) on Amazon Echo customer reviews.

    To clean and preprocess text data (removing punctuation, stopwords, etc.) from the reviews.

    To visualize the most frequent words in both positive and negative reviews using word clouds.

    To build and evaluate machine learning models (Naive Bayes, Logistic Regression, Gradient Boosting) for sentiment classification.

    To identify the strengths and weaknesses of each model in predicting customer feedback.

**Tools and Technologies**
    Python: The primary programming language used for data manipulation, analysis, and model building.

    Pandas: For data loading, manipulation, and analysis of the tabular review data.

    NumPy: For numerical operations, especially with array transformations.

    Matplotlib & Seaborn: For data visualization, including count plots and heatmaps for confusion matrices.

    WordCloud: For generating word cloud visualizations to identify frequent terms in text data.

    NLTK (Natural Language Toolkit): For text preprocessing tasks such as tokenization, removing stopwords, and handling punctuation.

    Scikit-learn: For machine learning model implementation, including:

        CountVectorizer for text feature extraction.

        MultinomialNB for Naive Bayes classification.

        LogisticRegression for logistic regression classification.

        GradientBoostingClassifier for gradient boosting classification.

        train_test_split for splitting data into training and testing sets.

        classification_report and confusion_matrix for model evaluation.

    Jupyter Notebooks (or Google Colab): For interactive development, code execution, and presenting the analysis steps and results.

**Key Steps**
    Data Loading and Initial Exploration:

        Load the amazon_reviews.csv dataset into a Pandas DataFrame.

        Perform initial data inspection using df.info() and df.describe().

    Feature Engineering:

        Convert the verified_reviews column to string type to ensure consistent text processing.

        Calculate the length of each review and add it as a new feature (length column).

    Exploratory Data Analysis (EDA):

        Visualize the distribution of ratings using a count plot.

        Plot a histogram of review lengths to understand review verbosity.

        Analyze the distribution of feedback (positive vs. negative) using a count plot.

    Text Preprocessing:

        Define a message_cleaning function to:

            Remove punctuation from review text.

            Remove common English stopwords.

        Apply the message_cleaning function to the verified_reviews column.

        Convert the cleaned text data into a numerical feature matrix using CountVectorizer.

    Sentiment Classification Model Building:

        Split the preprocessed data into training and testing sets (X_train, X_test, y_train, y_test).

        Train and evaluate three different classification models:

            Multinomial Naive Bayes (MNB)

            Logistic Regression

            Gradient Boosting Classifier

    Model Evaluation:

        For each model, generate and visualize a confusion matrix using seaborn.heatmap.

        Print a detailed classification_report showing precision, recall, F1-score, and support for each class (positive/negative feedback).

**Key Results**
    Data Overview: The dataset contains 3150 Amazon Echo reviews with columns for rating, date, variation, verified_reviews, and feedback.

    Sentiment Distribution: A significant majority of reviews are positive (feedback = 1, approximately 2893 reviews), while a smaller portion are negative (feedback = 0, approximately 257 reviews).

    Word Clouds: Word clouds for positive reviews typically highlight terms like "love," "great," "music," "easy," and "sound." Negative review word clouds emphasize issues such as "sound," "work," "problem," and "connect."

    Model Performance:

        Multinomial Naive Bayes: Achieved an accuracy of around 93%. It performed well in recalling positive reviews (0.97 recall for class 1) but had lower precision (0.53) and recall (0.39) for negative reviews, indicating difficulty in accurately identifying all negative instances.

        Logistic Regression: Showed a higher overall accuracy of 96%. It significantly improved precision for negative reviews (0.90) compared to Naive Bayes, while maintaining high performance for positive reviews. Recall for negative reviews was 0.43, suggesting it still misses some negative cases but is more precise when it does predict a negative sentiment.

        Gradient Boosting Classifier: Achieved an accuracy of 94%. Similar to Logistic Regression, it had good precision for negative reviews (0.64) but lower recall (0.20), implying it's very conservative in predicting negative sentiment, missing most of them. It excelled in classifying positive reviews (0.99 recall).

**Steps to Reproduce**
    Clone the Repository:

    git clone <repository_url>
    cd amazon-echo-reviews-analysis

    (Replace <repository_url> with the actual URL of your repository if applicable).

    Prepare the Dataset:

    Ensure the amazon_reviews.csv file is in the root directory of the project. If not, you may need to upload it.

    Install Dependencies:

    pip install pandas numpy matplotlib seaborn wordcloud scikit-learn nltk

    Download NLTK Data:

        Run the following Python commands in your environment (e.g., Jupyter Notebook, Colab, or a Python script) to download necessary NLTK data:

        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

    Run the Jupyter Notebook (or Google Colab Notebook):

        Open the amazon_echo_reviews_analysis.ipynb file in a Jupyter environment or Google Colab.

        Execute all cells sequentially to reproduce the analysis, visualizations, and model training/evaluation.
