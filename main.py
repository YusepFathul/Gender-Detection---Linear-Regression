# %% [markdown]
# ## Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re

# %% [markdown]
# ## Data Understanding

# %%
df = pd.read_csv("data-pemilih-kpu.csv")  
print("Dataset Shape:", df.shape)
df

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
plt.figure(figsize=(14,7))  
sns.countplot(x=df['jenis_kelamin'], palette="viridis")
plt.title("Gender Distribution", fontsize=14)
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Count", fontsize=12)

plt.xticks(rotation=0, ha='center')  #
plt.show()

# %% [markdown]
# ## Data Preprocessing

# %%
df.isnull().sum()
df = df.dropna()

# %%
def clean_name(name):
    return re.sub(r'[^a-zA-Z]', ' ', name).lower().strip()

df['nama'] = df['nama'].apply(clean_name)

df.isnull().sum()

# %%
label_encoder = LabelEncoder()
df['jenis_kelamin'] = label_encoder.fit_transform(df['jenis_kelamin'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))  
print("Label Mapping:", label_mapping)

# %% [markdown]
# ## Feature Engineering

# %%
X = df['nama']
y = df['jenis_kelamin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
sns.countplot(x=y_train, label='Train Data', color='blue', alpha=0.6)
sns.countplot(x=y_test, label='Test Data', color='red', alpha=0.6)
plt.xlabel("Gender", fontsize=12)
plt.legend()
plt.title('Distribution of Training and Test Data')
plt.show()

# %%
# Vectorization with CountVectorizer using n-gram characters
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2,6))
vectorizer.fit(X_train.ravel())

X_train = vectorizer.transform(X_train.ravel())
X_test = vectorizer.transform(X_test.ravel())

# %% [markdown]
# ## Modelling

# %%
model = LogisticRegression(max_iter=500, solver='liblinear')
model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluation Model

# %%
train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# Evaluation model
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# %%
# Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# ## Usage Example

# %%
def prediksi_jenis_kelamin(nama):
    nama = clean_name(nama)
    return label_encoder.inverse_transform([model.predict(vectorizer.transform([nama]))[0]])[0]
    return label_mapping[prediksi]

print(prediksi_jenis_kelamin("Yusep"))


