# Classification_models_on_MNIST_dataset
A study on how different classification models performed on MNIST dataset which consists of handwritten digits from 0 to 9

# Objective

The objective of this repository is to compare the performance of various popular classification models on a given dataset. We aim to evaluate these models based on key performance metrics: Accuracy, Precision, Recall, and F1-Score. This repository helps in understanding the strengths and weaknesses of each model and guides the selection of an appropriate model for similar classification problems.

# Classification Models Overview

**1. Logistic Regression**

Logistic Regression is a statistical model that uses a logistic function to model a binary dependent variable. It is simple and interpretable, suitable for linearly separable data.

**2. Support Vector Machine (SVM)**

SVM finds the optimal hyperplane that best separates the classes in the feature space. It is effective in high-dimensional spaces and uses kernel tricks for non-linear classification.

**3. K-Nearest Neighbors (KNN)**

KNN is a non-parametric, instance-based learning algorithm. It classifies new data points based on the majority label among the 'k' closest training examples.

**4. Naive Bayes**

Naive Bayes is a probabilistic classifier based on Bayes' Theorem with a strong (naive) independence assumption between the features. It performs well on high-dimensional datasets.

**5. Decision Tree**

Decision Tree builds a model in the form of a tree structure. It splits the data into subsets based on the value of input features, making decisions based on entropy or Gini index.

**6. Random Forest**

Random Forest is an ensemble method that constructs multiple decision trees and merges them to get more accurate and stable predictions. It reduces overfitting and improves generalization.

# Model Performance Comparison



# Key Insights

**Logistic Regression:** Performs well with linearly separable data; fast and easy to interpret.

**SVM:** Offers high accuracy, especially with kernel trick; computationally intensive for large datasets.

**KNN:** Simple to implement; performance heavily depends on choice of 'k' and distance metric.

**Naive Bayes:** Surprisingly effective despite naive assumptions; best for text classification tasks.

**Decision Tree:** Easy to visualize and interpret; prone to overfitting.

**Random Forest:** Best performer overall; robust against overfitting and handles non-linear data effectively.

# Conclusion

Each classification model has its unique strengths and is suited to different types of datasets and problems. Random Forest showed the best overall performance in our case, making it a strong default choice. However, depending on the specific use case and computational resources, other models like SVM or Logistic Regression might be preferred for their simplicity or speed.
