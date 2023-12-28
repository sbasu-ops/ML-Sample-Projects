# ML-Sample-Projects

This repository houses sample machine learning and data science projects. A high level summary of the projects is provided below

1. LSTM based Ecommerce Review Sentiment Analysis: This project centers around a sentiment analysis system built on LSTM (Long Short-Term Memory) architecture using the Kaggle dataset available at: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews/notebooks.
  Key Highlights:

Exploratory Data Analysis: The accompanying notebook includes a concise exploratory data analysis focusing on word distribution within the dataset, providing insights into the underlying structure of the reviews.

Model Architecture: Utilizing a bidirectional LSTM model equipped with Glove embeddings and an early stopping mechanism, the system analyzes sentiments embedded within recommendations and reviews from a prominent e-commerce platform, specifically targeting women's clothing.

Performance Metrics:

Accuracy: The model achieves an 87% accuracy rate in correctly predicting recommendations and a 93% accuracy rate in discerning the positive or negative sentiment conveyed within the reviews.

Comprehensive Analysis: Beyond accuracy, the project delves into metrics such as precision, recall, ROC-AUC, and employs a confusion matrix to provide a holistic assessment of the model's performance.

Insights and Potential Enhancements:

While the model demonstrates notable proficiency in predicting positive reviews, the predominant class within the dataset, there exists room for advancement in effectively analyzing reviews of a mixed or neutral sentiment. This identifies an area for potential enhancement and further fine-tuning of the model.


2. Network Alpha: Stock Similarity Visualization using D3.js, Bloomberg API, and k-NN Algorithm
This project features a visualization tool built with D3.js that identifies stocks sharing similarities in fundamental metrics. Leveraging data obtained through the Bloomberg API and employing the k-Nearest Neighbors (k-NN) algorithm, this tool constructs a graph visualization where nearest neighbor stocks are interconnected.

Key Features:

Adjustable Nearest Neighbors: Users can modify the number of nearest neighbors to enhance the similarity of stock returns, focusing on a five-year time span.

User Input Functionality: Users have the capability to input a specific stock, enabling the identification of its five closest neighbors. This functionality provides a comparative analysis of stock returns, visually represented through a customizable bar chart.

This visualization tool aims to assist users in understanding stock relationships based on fundamental metrics, facilitating comparative analysis and insights into stock performance over time.
