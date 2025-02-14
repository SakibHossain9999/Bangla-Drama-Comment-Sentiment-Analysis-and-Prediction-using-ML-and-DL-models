# Bangla-Drama-Comment-Sentiment-Analysis-and-Prediction-using-ML-and-DL-models

This project applies machine learning and deep learning models to classify Bangla drama comments into positive and negative sentiments. The project leverages Natural Language Processing (NLP) techniques and a supervised learning approach to preprocess, train, and evaluate multiple machine learning models on a custom dataset.

## Project Overview

The repository contains Python code for data preprocessing, feature extraction, model training, evaluation, and prediction. Using a Logistic Regression model, the project provides insights into comment sentiments, specifically for Bangla dramas, by classifying comments as either positive or negative.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation](#evaluation)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)

## Dataset

Dataset Source: [**Bengali Sentiment Classification**](https://www.kaggle.com/datasets/saurabhshahane/bengali-sentiment-classification)

Description:
The dataset consists of two text files:
- **`all_positive_8500.txt`**: Contains positive comments
- **`all_negative_3307.txt`**: Contains negative comments

Each line in these files represents a single comment. These comments were combined, labeled, and saved as `dataset.csv` for training and testing purposes.

## Preprocessing

To prepare the comments for machine learning, the following preprocessing steps were applied:

1. **Text Cleaning**: Removed non-alphabet characters and extra whitespaces.
2. **Stop Words Removal**: Removed Bengali stop words using the NLTK library.
3. **Label Encoding**: Encoded labels as `1` (positive) and `0` (negative).

To prepare the comments for Deep Learning, the following preprocessing steps were applied:
1. **Text Cleaning**: Removed urls, html tags, punctuations(like commas, periods, etc.), numbers, extra whitespaces, convert all the text to lowercase, and non-Bengali alphabet characters.
2. **Unicode Normalization**: Applies Bengali-specific Unicode normalization, though it is commented out in this code. If used, it would standardize the Bengali script characters for consistent encoding.
3. **Correct Common Spelling Variations**: Replaces specific spelling variations (an example here, replaces "ব্যাপক" with "বেশি") to maintain consistency in wording.
4. **Stop Words Removal**: Removed Bengali stop words using the NLTK library.
5. **Rejoin Tokens**: Combines the cleaned tokens back into a single string.
6. **Label Encoding**: Encoded labels as `1` (positive) and `0` (negative).
7. **Tokenize Text for Model Training**: Converts text into sequences of integers, where each integer represents a token.
8. **Padding Sequences**: Pads sequences to ensure they have a uniform length for model input.
9. **Convert Labels to Categorical Format**: Converts the labels into a one-hot encoded format suitable for classification.

## Feature Extraction

To transform the comments into a numerical format for machine learning:
- **TF-IDF Vectorizer**: A TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer was used to represent comments as weighted word vectors.

## Machine Learning Models

The following models were tested on the processed data:
- **Logistic Regression** (selected for final deployment)
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Passive Aggressive Classifier**

## Deep Learning Model
- **Hyperparametertuned RNN**

## Evaluation

The models were evaluated on the test set using:
- **Accuracy Score**: To measure the overall performance.
- **Classification Report**: Including precision, recall, and F1-score.
- **Confusion Matrix**: To analyze the distribution of true positives, false positives, etc.

The Logistic Regression model provided the best balance of performance, and therefore was saved for further predictions.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Bangla-Drama-Comment-Sentiment-Analysis.git
    cd Bangla-Drama-Comment-Sentiment-Analysis-and-Prediction
    ```

2. **Install Dependencies**:
    Install the required Python packages by running:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the .ipynb of any machine or deep learning model(s)**

## Licensing

**This project is licensed under the MIT License, a permissive open-source license that allows others to use, modify, and distribute the project's code with very few restrictions. This license can benefit research by promoting collaboration and encouraging the sharing of ideas and knowledge. With this license, researchers can build on existing code to create new tools, experiments, or projects, and easily adapt and customize the code to suit their specific research needs without worrying about legal implications. The open-source nature of the MIT License can help foster a collaborative research community, leading to faster innovation and progress in their respective fields. Additionally, the license can help increase the visibility and adoption of the project, attracting more researchers to use and contribute to it.**
