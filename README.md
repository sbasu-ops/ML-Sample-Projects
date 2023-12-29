# ML-Sample-Projects

This repository houses sample machine learning and data science projects. A high level summary of the projects is provided below

1. **LSTM based Ecommerce Review Sentiment Analysis:**
   
This project centers around a sentiment analysis system built on LSTM (Long Short-Term Memory) architecture using the Kaggle dataset available at: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews/notebooks.
Key Highlights:

Exploratory Data Analysis: The accompanying notebook includes a concise exploratory data analysis focusing on word distribution within the dataset, providing insights into the underlying structure of the reviews.

Model Architecture: Utilizing a bidirectional LSTM model equipped with Glove embeddings and an early stopping mechanism, the system analyzes sentiments embedded within recommendations and reviews from a prominent e-commerce platform, specifically targeting women's clothing.

Performance Metrics:

Accuracy: The model achieves an 87% accuracy rate in correctly predicting recommendations and a 93% accuracy rate in discerning the positive or negative sentiment conveyed within the reviews.

Comprehensive Analysis: Beyond accuracy, the project delves into metrics such as precision, recall, ROC-AUC, and employs a confusion matrix to provide a holistic assessment of the model's performance.

Insights and Potential Enhancements:

While the model demonstrates notable proficiency in predicting positive reviews, the predominant class within the dataset, there exists room for advancement in effectively analyzing reviews of a mixed or neutral sentiment. This identifies an area for potential enhancement and further fine-tuning of the model.


2. **Network Alpha: Stock Similarity Visualization using D3.js, SEC EDGAR, and k-NN Algorithm**
   
This project features a visualization tool built with D3.js that identifies stocks sharing similarities in fundamental metrics. Leveraging data obtained from the SEC EDGAR database and employing the k-Nearest Neighbors (k-NN) algorithm, this tool constructs a graph visualization where nearest neighbor stocks are interconnected.

Key Features:

Adjustable Nearest Neighbors: Users can modify the number of nearest neighbors to enhance the similarity of stock returns, focusing on a five-year time span.

User Input Functionality: Users have the capability to input a specific stock, enabling the identification of its five closest neighbors. This functionality provides a comparative analysis of stock returns, visually represented through a customizable bar chart.

This visualization tool aims to assist users in understanding stock relationships based on fundamental metrics, facilitating comparative analysis and insights into stock performance over time.

3. **Multiclass tumor classification based on RNA-Seq Gene Expression Data**
   
Explored various dimensionality reduction techniques and classification algorithms to differentiate between five distinct tumor classes within a dataset comprising 801 samples and 20,531 gene expression dimensions sourced from the UCI Machine Learning Repository.

Classification Algorithms:The study encompassed several classification methodologies: k-Nearest Neighbors (k-NN), Linear Discriminant Analysis (LDA), Naive Bayes, Support Vector Machine (both Linear and RBF Kernel), Logistic Regression (Elastic Net), Decision Tree, Random Forest, and AdaBoost.

Dimensional Reduction: Feature selection involved ANOVA F-test and Mutual Information classification, while dimensional reduction techniques included Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA). 

Results: All classification approaches achieved high accuracy, precision, and recall for tumor class identification. Notably, every method demonstrated an accuracy exceeding 97%, with the lowest F1-score at 0.88. The majority of methods exhibited F1-scores surpassing 0.98, signifying robust predictive capabilities.


4. **RobustNet : Adversarial Training and Data Augmentation Techniques against ImageNetC Corruptions.**
   
CNNs often struggle with domain transfer and handling out-of-distribution data encountered in real-world applications, differing from meticulously curated training datasets. This stands in stark contrast to the robustness of human vision systems, which remain resilient to such input variations and are unaffected by minor alterations. This resilience in humans might stem from their capacity to grasp global features, unlike CNNs, which lean towards texture-based cues. Despite this disparity, as deep learning-based vision systems find widespread application in domains like autonomous vehicles, robotics, and industrial settings, ensuring their resilience against common corruptions (such as image noise, digital alterations, weather-related issues, and blurred images) becomes paramount.

RobustNet introduces perturbation-based adversarial training, creating misleading images, and employs data augmentation techniques (including style transfer, AugMix, random cropping, and horizontal flipping) to enhance the corruption resilience of a CNN, specifically a ResNet-18 model. A subset of 20 classes from ImageNet is utilized for adversarial training and augmentations, with performance evaluation conducted using Mean Corruption Error (mCE) on the ImageNet-C dataset containing the same 20 classes but with random corruptions.
The findings highlight that fine-tuning a pre-trained ResNet-18 on deliberately misclassified perturbation-based misleading images, in conjunction with clean data, yields the best performance with an mCE of 0.76. Encouragingly, similar enhancements are observed when training ResNet-18 from scratch, illustrating its applicability beyond pre-trained models. Evaluating data augmentation techniques revealed that AugMix outperforms others (mCE = 0.96), followed by the stylized ImageNet method (mCE = 1.01). A combined model trained from scratch on Stylized Images and ImageNet20 with AugMix achieves an mCE of 0.91. Several different analysis methods such as kernel weights and maps visualization, Centered Kernel Alignment are used to understand the mCE improvements across different approaches.
