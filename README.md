# Classification_models_on_MNIST_dataset
A study on how different classification models performed on MNIST dataset which consists of handwritten digits from 0 to 9

# Objective

The objective of this repository is to compare the performance of various popular classification models on the MNIST dataset, which contains 70,000 images of handwritten digits (0–9). We aim to evaluate these models based on key performance metrics: Accuracy, Precision, Recall, and F1-Score. This repository helps in understanding the strengths and weaknesses of each model and guides the selection of an appropriate model for image-based classification problems.

# Dataset

The dataset used in this project is the [MNIST in CSV format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) available on Kaggle. It is a version of the classic MNIST dataset converted into CSV format for easy loading and analysis using common data processing libraries like pandas.

Training set: mnist_train.csv (60,000 samples)

Test set: mnist_test.csv (10,000 samples)

Each sample represents a 28x28 grayscale image of a handwritten digit (0 through 9), flattened into a 784-length vector. The first column in each row is the label (the digit), and the remaining 784 columns represent pixel intensity values (0–255). This format allows for quick integration with machine learning workflows without requiring image preprocessing or specialized libraries for image handling.

# Classification Models Overview

**1. Logistic Regression**

Logistic Regression is a statistical model that uses a logistic function to model a binary dependent variable. Despite its simplicity, it can be extended to multiclass problems like MNIST and performs surprisingly well on image data after flattening and normalization.

**2. Kernel Support Vector Machine (Kernel SVM)**

Kernel SVM finds the optimal hyperplane that best separates the classes in the feature space using kernel functions. It is effective in high-dimensional spaces and handles the pixel-based features of MNIST well. However, it is computationally intensive on large datasets.

**3. K-Nearest Neighbors (KNN)**

KNN is a non-parametric, instance-based learning algorithm. It classifies new data points based on the majority label among the 'k' closest training examples. While intuitive and easy to understand, its performance on MNIST benefits significantly from appropriate scaling and choice of distance metric. With optimal tuning, KNN achieves high accuracy, but inference time and memory consumption increase with data size.

**4. Naive Bayes**

Naive Bayes is a probabilistic classifier based on Bayes' Theorem with a strong (naive) independence assumption between the features. On MNIST, it is limited by its assumptions, but still provides a useful baseline.

**5. Decision Tree**

Decision Tree builds a model in the form of a tree structure. It splits the data into subsets based on the value of input features, making decisions based on entropy or Gini index. While interpretable, it struggles with the complexity of image data in MNIST.

**6. Random Forest**

Random Forest is an ensemble method that constructs multiple decision trees and merges them to get more accurate and stable predictions. It handles the complexity of MNIST better than a single decision tree and improves generalization.

# Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.92     | 0.92      | 0.92   | 0.92     |
| Kernel SVM          | 0.97     | 0.97      | 0.97   | 0.97     |
| KNN                 | 0.94     | 0.94      | 0.94   | 0.94     |
| Naive Bayes         | 0.52     | 0.68      | 0.52   | 0.48     |
| Decision Tree       | 0.89     | 0.89      | 0.89   | 0.89     |
| Random Forest       | 0.97     | 0.97      | 0.97   | 0.97     |


# Key Insights

**Logistic Regression:** Performs surprisingly well on the MNIST dataset despite its simplicity. A good baseline for image classification.

**Kernel SVM:** Delivers high accuracy on MNIST using kernel tricks to manage non-linearity; computationally demanding for large image datasets.

**KNN:** Achieves excellent accuracy when properly tuned and normalized. However, it is computationally inefficient at scale, making it better suited for smaller datasets or offline predictions.

**Naive Bayes:** Limited by its independence assumption; not ideal for image data like MNIST but useful for quick benchmarks.

**Decision Tree:** Struggles with the complexity of MNIST digits; prone to overfitting without pruning.

**Random Forest:** Consistently strong performance on MNIST due to ensemble learning; good balance between accuracy and robustness.

# Conclusion

This comparative analysis of classification models on the MNIST dataset highlights the diverse strengths and trade-offs of each approach:

**Kernel SVM** and **Random Forest** provided the highest overall performance across all key metrics, making them excellent choices when accuracy is the priority.

**KNN** also delivered strong results but is less practical for large-scale or real-time applications due to its high inference cost.

**Logistic Regression** stood out as a robust and interpretable baseline model with respectable accuracy and minimal computational demands.

**Decision Tree** offered decent performance and interpretability but lagged behind ensemble methods.

**Naive Bayes**, while fast and simple, showed limited performance due to its assumptions, especially on complex, high-dimensional image data.

The ideal model choice depends on the specific requirements of your task, including interpretability, computational resources, and prediction latency.
