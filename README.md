# ML-Sample-Projects

This repository houses sample machine learning and data science projects. A high level summary of the projects is provided below

1. LSTM based Ecommerce Review Sentiment Analysis: The project is based on the Kaggle dataset: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews/notebooks
A brief exploratory data analysis focused on distribution of words is presented in the notebook. A bidirectional LSTM model with early stopping and Glove embeddings is used to analyze the sentiment of recommendations and reviews on Womens clothing taken from a popular ecommerce website. The model predicts the 
correct recommendation 87% and the positive or negative sentiment of the reviews 93% of the time. Other metrics like precistion, recall, ROC-AUC, confusion matrix are also analyzed.
The model performs very well in prediction postive reviews, which is the majority class. It has scope of further improvement in analyzing mixed or neutral reviews. 
