# Credit Card Fraud Detection

This project implements credit card fraud detection using machine learning techniques, specifically Decision Tree, K-Means clustering, and K-Nearest Neighbors (KNN) algorithms.

## Project Overview

The goal of this project is to detect fraudulent credit card transactions using different classification approaches:

1. Decision Tree Classifier
2. K-Means Clustering
3. K-Nearest Neighbors (KNN) Classifier

## Dataset

The project uses a credit card transaction dataset (`creditcard.csv`) with the following characteristics:

- Total transactions: 284,807
- Fraudulent transactions: 492 (0.17% of total)
- Features: Time, V1-V28 (anonymized features), Amount, Class (0 for legitimate, 1 for fraudulent)

## Requirements

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib (for visualizations, if needed)

Install the required libraries using:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Implementation

### 1. Decision Tree Classifier

**File:** `ML_PROJECT_DT.ipynb`

- Loads and preprocesses the dataset
- Splits data into training and testing sets
- Trains a Decision Tree Classifier
- Evaluates the model using accuracy and confusion matrix

### 2. K-Means Clustering

**File:** `ML_PROJECT_KMEANS.ipynb`

- Loads and preprocesses the dataset
- Normalizes features
- Applies K-Means clustering (k=2)
- Evaluates the clustering results using accuracy, precision, recall, and F1 score

### 3. K-Nearest Neighbors (KNN) Classifier

**File:** `ML_PROJECT_KNN.ipynb`

- Implements KNN using `KNeighborsClassifier`
- Trains the model on the preprocessed dataset
- Evaluates the model using accuracy, precision, recall, and F1 score

## Results

### Decision Tree Classifier

- Accuracy: 99.93%
- Confusion Matrix:
  ```
  [[42644    14]
   [   17    47]]
  ```

### K-Means Clustering

- Accuracy: 79.61%
- Precision: 0.0046
- Recall: 0.5294
- F1 Score: 0.0092
- Confusion Matrix:
  ```
  [[22648  5782]
   [   24    27]]
  ```

### K-Nearest Neighbors (KNN) Classifier

- Accuracy: 94.92%
- Precision: 0.98
- Recall: 0.92
- F1 Score: 0.95

## Usage

1. Ensure you have the required libraries installed.
2. Place the `creditcard.csv` file in the same directory as the Jupyter notebooks.
3. Run the Jupyter notebooks:
   - `ML_PROJECT_DT.ipynb` for Decision Tree classification
   - `ML_PROJECT_KMEANS.ipynb` for K-Means clustering
   - `ML_PROJECT_KNN.ipynb` for KNN classification

## Conclusion

The Decision Tree Classifier shows the best performance in detecting fraudulent transactions, followed by the K-Nearest Neighbors Classifier. The K-Means clustering approach performs relatively poorly due to its unsupervised nature.
