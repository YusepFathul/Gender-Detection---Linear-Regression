# Gender Detection - Yusep Fathul Anwar

## Project Overview

In this project, I built a machine learning model to predict gender based on names. This model can be used in various applications such as demographic surveys, social media analysis, and gender-based personalization services.

In previous research, NLP-based models have been used to classify names based on linguistic patterns. The dataset used in this project was obtained from the KPU voter data, which includes name and gender information.

## Business Understanding

### Problem Statements

1. How can machine learning classify gender based on names?
2. How accurate is the linear regression model in performing gender classification?
3. How can model performance be improved through better data preprocessing?

### Goals

1. Develop a machine learning model to classify gender based on patterns in names.
2. Evaluate the model's performance using appropriate metrics to ensure prediction accuracy.
3. Optimize data preprocessing so that the model can better classify gender.

### Solution Approach

1. Using **Linear Regression** as the primary model for gender prediction.
2. Performing **data preprocessing**, including normalizing name text to remove non-alphabet characters.
3. Applying **vectorization using CountVectorizer** to convert text data into a numerical representation usable by the model.

## Data Understanding

The dataset used in this project comes from KPU voter data, which includes the following information:
<p align="center">
  <img src="https://github.com/user-attachments/assets/f21a5ea0-7ba7-45b5-9c98-d270304d6366" width="600">
</p>

- **nama**: The full name of an individual.
- **jenis_kelamin**: The target label indicating whether the individual is male or female.

The gender distribution was analyzed using visualization to ensure data balance before training the model.
<p align="center">
  <img src="https://github.com/user-attachments/assets/5015c452-970c-4911-bcc1-c7b6db7199dd" width="750">
</p>

## Data Preparation

Before training the model, several preprocessing steps were performed, including:

- Removing non-alphabet characters from names.
- Converting text to lowercase for consistency.
- Using **CountVectorizer** to convert text into numerical features.
- Splitting the dataset into **training set (80%)** and **testing set (20%)** for model evaluation.

  <p align="center">
  <img src="https://github.com/user-attachments/assets/8fffc494-6862-4556-a2d6-66aba7bb9e1e" width="600">
</p>


## Modeling

The model used in this project is **Linear Regression**. The training process was conducted through the following steps:

1. Training the model using preprocessed training data.
2. Using transformed text features as input for the model.
3. Using the target label (gender) as the dependent variable.
   
   <p align="center">
  <img src="https://github.com/user-attachments/assets/c828e727-787a-4f4b-892d-964366f471b2" width="600">
</p>

## Evaluation

The model was evaluated using **accuracy score**, **confusion matrix**, and **classification report**. This evaluation aimed to determine how well the model can classify gender based on names.

### Evaluation Results

- **Training Accuracy:** 0.9963 (99.63%)
- **Testing Accuracy:** 0.9355 (93.55%)
- **Classification Report:**

  | Class | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | Female (0) | 0.93 | 0.94 | 0.93 | 1232 |
  | Male (1) | 0.94 | 0.93 | 0.94 | 1358 |
  | **Accuracy** | | | 0.94 | 2590 |
  | **Macro Avg** | 0.94 | 0.94 | 0.94 | 2590 |
  | **Weighted Avg** | 0.94 | 0.94 | 0.94 | 2590 |


## Conclusion

This project successfully built a machine learning model to classify gender based on names. Linear Regression was used in this experiment, with evaluation results showing good performance. Future steps could involve exploring more complex models if needed to improve prediction accuracy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b726ea8-eaa4-407a-9c63-cf4ef34e893c" width="600">
</p>

