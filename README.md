# Gender Detection - Yusep Fathul Anwar

## Project Overview

This project focuses on building a machine learning model to predict gender based on names. The model can be applied in various domains such as demographic surveys, social media analysis, and gender-based personalization services. The dataset utilized in this project was obtained from the KPU voter data, which includes name and gender information.

## Business Understanding

### Problem Statements

1. How can machine learning classify gender based on names?
2. How accurate is the logistic regression model in performing gender classification?
3. How can model performance be improved through better data preprocessing?

### Goals

1. Develop a machine learning model to classify gender based on patterns in names.
2. Evaluate the model's performance using appropriate metrics to ensure prediction accuracy.
3. Optimize data preprocessing to enhance model performance.

### Solution Approach

1. Using **Logistic Regression** as the primary model for gender prediction.
2. Performing **data preprocessing**, including normalizing name text to remove non-alphabet characters.
3. Applying **vectorization using CountVectorizer** to convert text data into a numerical representation usable by the model.

## Data Understanding

The dataset used in this project comes from KPU voter data, with the following structure:

- **nama**: The full name of an individual.
- **jenis_kelamin**: The target label indicating whether the individual is male or female.

### Data Summary

- **Number of rows:** 13137
- **Number of columns:** 2
- **Missing Values:** Name:187, Gender:187
- **Outliers:** Not applicable due to categorical nature of the data.

### Dataset Source

The dataset was obtained from the KPU's official voter data records.

### Gender Distribution

The gender distribution was analyzed to ensure class balance before training the model:

- **Female:** 52.42%
- **Male:** 47.58%

The balanced distribution helps prevent bias in the model.

**Gender Distribution Visualization:**

![Gender Distribution](https://github.com/user-attachments/assets/5015c452-970c-4911-bcc1-c7b6db7199dd)

## Data Preparation

The following steps were performed to prepare the dataset for modeling:

1. **Handling Missing Values**: Missing values were handled by using the `dropna()` function to remove incomplete records.
2. **Removing Non-Alphabet Characters**: Special characters and digits were removed from the names using a regex pattern.
3. **Encoding Target Variable**: The gender variable was encoded into binary values (0 for Female, 1 for Male) using `LabelEncoder`.
4. **Dataset Splitting**: The dataset was split into:
   - **Training Set**: 80% of the data
   - **Testing Set**: 20% of the data
5. **Text Vectorization**: Used **CountVectorizer** with `char_wb` analyzer and n-grams of length 2-6 to capture name patterns.

**Data Preparation Steps Visualization:**

  <p align="center">
  <img src="https://github.com/user-attachments/assets/8fffc494-6862-4556-a2d6-66aba7bb9e1e" alt="Data Preparation">
</p>

## Modeling

The model utilized in this project is **Logistic Regression**. This algorithm was chosen due to its simplicity and effectiveness in binary classification tasks involving text data.

### Model Development

- **Algorithm:** Logistic Regression  
- **Parameters:** max_iter=500, solver='libLogistic'.
- **Feature Extraction:** CountVectorizer with `char_wb` analyzer and n-grams (2,6) to capture name patterns.

### Model Training Process

1. The model was trained on the training dataset using the preprocessed text features as input.
2. The target variable (gender) was used as the dependent variable.
3. The model's performance was evaluated on the testing set.

**Model Training Visualization:**
<p align="center">
  <img src="https://github.com/user-attachments/assets/c828e727-787a-4f4b-892d-964366f471b2" >
</p>


## Evaluation

The model was evaluated using **accuracy score**, **confusion matrix**, and **classification report** to measure classification performance.

### Evaluation Results

- **Training Accuracy:** 99.63%
- **Testing Accuracy:** 93.55%

**Classification Report:**

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Female (0)       | 0.93      | 0.94   | 0.93     | 1232    |
| Male (1)         | 0.94      | 0.93   | 0.94     | 1358    |
| **Accuracy**     |           |        | 0.94     | 2590    |
| **Macro Avg**    | 0.94      | 0.94   | 0.94     | 2590    |
| **Weighted Avg** | 0.94      | 0.94   | 0.94     | 2590    |

**Confusion Matrix Visualization:**
<p align="center">
  <img src="https://github.com/user-attachments/assets/6aff9f8a-6821-4cc9-a52e-8258a8576e2d" >
</p>


### Analysis

The model achieved high accuracy, with relatively balanced precision, recall, and F1-scores for both classes. Misclassifications may arise due to ambiguous names commonly associated with both genders.

## Conclusion

This project successfully developed a machine learning model for gender detection based on names. The use of Logistic Regression resulted in strong performance metrics, particularly with careful data preprocessing and feature engineering.



