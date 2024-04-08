# Spam Message Detection Project

## Overview

This project aims to detect spam messages using various machine learning and deep learning techniques. We'll explore different models including Naive Bayes, Support Vector Machines (SVM), Recurrent Neural Networks (RNN), and Convolutional Neural Networks (CNN) to classify messages as spam or not spam.

## Dataset

The dataset used in this project is the "SPAM text message 20170820" dataset, which contains a collection of SMS messages labeled as spam or ham (not spam). The dataset is publicly available and can be found [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

## Data Preprocessing

Before building the models, we preprocess the text data by converting it to lowercase, removing punctuation, and tokenizing the text. We also split the dataset into training and testing sets for model evaluation.

## Models

### Naive Bayes and SVM

We start by using traditional machine learning algorithms such as Naive Bayes and SVM to classify the messages based on their bag-of-words representation. We use the `CountVectorizer` to convert text messages into numerical features and train the classifiers on the training data. Then, we evaluate their performance on the testing data using accuracy as the metric.

### Recurrent Neural Network (RNN)

Next, we explore the use of Recurrent Neural Networks (RNNs) for text classification. We use a Long Short-Term Memory (LSTM) network to learn the sequential nature of the text data. The messages are tokenized and padded sequences are used as input to the LSTM model. We train the model on the training data and evaluate its performance on the testing data.

### Convolutional Neural Network (CNN)

We also experiment with Convolutional Neural Networks (CNNs) for text classification. We use a 1D CNN architecture with an embedding layer followed by convolutional and pooling layers. The model learns to extract local features from the text data and classify messages as spam or not spam. Similar to the RNN model, we train the CNN on the training data and evaluate its performance on the testing data.

## Results

We compare the performance of different models based on their accuracy on the testing data. Additionally, we visualize the confusion matrices to understand the classification performance in more detail. The results provide insights into the effectiveness of each approach for spam message detection.

## Conclusion

In conclusion, this project demonstrates various machine learning and deep learning techniques for spam message detection. By comparing different models, we can identify the most suitable approach for accurately classifying spam messages and reducing unwanted communication.

For more details and code implementation, refer to the Jupyter Notebooks provided in this repository.

