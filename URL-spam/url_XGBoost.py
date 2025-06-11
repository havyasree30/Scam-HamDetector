import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import nlpaug.augmenter.word as naw
import xgboost as xgb

# **1️⃣ Set Seed for Reproducibility**
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# **2️⃣ Download Stopwords**
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# **3️⃣ Load Dataset with Detailed Debugging**
data = []
skipped_lines = []
try:
    with open('dataset_phishing.csv', encoding='utf-8') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        start_idx = 1  # Skip header
        for i, line in enumerate(lines[start_idx:], start=start_idx):
            if line.strip():  # Skip empty lines
                fields = line.strip().split(',')
                if len(fields) >= 2:  # Ensure at least URL and label
                    url = fields[0].strip()  # First column: url
                    label = fields[-1].strip()  # Last column: status
                    data.append([url, label])
                else:
                    skipped_lines.append((i+1, line.strip(), len(fields)))
                    print(f"Skipping line {i+1}: Insufficient fields ({len(fields)}) - Content: '{line.strip()}'")
            else:
                skipped_lines.append((i+1, "", 0))
                print(f"Skipping line {i+1}: Empty line")
except FileNotFoundError:
    print("Error: 'dataset_phishing.csv' not found. Please ensure the file is in the working directory.")

df = pd.DataFrame(data, columns=['url', 'label'])
df.dropna(subset=['url'], inplace=True)
print(f"Loaded {len(df)} rows from dataset")
print(f"Total skipped lines: {len(skipped_lines)}")
print("Sample of skipped lines (first 5):", skipped_lines[:5])

# **4️⃣ Text Cleaning Function**
def clean_text(text):
    text = re.sub(r'http[s]?://', '', text)  # Remove protocol
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s./]', '', text)  # Remove punctuation except dots and slashes
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_text'] = df['url'].apply(clean_text)

# **5️⃣ Label Encoding**
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # "legitimate" → 0, "phishing" → 1
print(f"Original legitimate count: {len(df[df['label'] == 0])}")
print(f"Original phishing count: {len(df[df['label'] == 1])}")

# **6️⃣ Balance Data: Equal Legitimate & Phishing Distribution**
df_legitimate = df[df['label'] == 0]
df_phishing = df[df['label'] == 1]

if len(df_legitimate) == 0 or len(df_phishing) == 0:
    print("Error: One class is empty. Cannot balance dataset.")
else:
    target_count = max(len(df_legitimate), len(df_phishing))
    df_legitimate_downsampled = resample(df_legitimate, replace=False, n_samples=target_count, random_state=42)
    df_phishing_oversampled = resample(df_phishing, replace=True, n_samples=target_count, random_state=42)
    df_balanced = pd.concat([df_legitimate_downsampled, df_phishing_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced dataset size: {len(df_balanced)}")

# **7️⃣ Data Augmentation (Increase Dataset to 100,000 Samples)**
augmentor = naw.SynonymAug(aug_src='wordnet')

def augment_text(text):
    try:
        return augmentor.augment(text)[0]  # Replace words with synonyms
    except:
        return text  # Return original text if error occurs

new_samples = []
while len(df_balanced) + len(new_samples) < 100000:  # Adjusted to 100K for larger size
    sample = df_balanced.sample(1).iloc[0].copy()
    sample['cleaned_text'] = augment_text(sample['cleaned_text'])
    new_samples.append(sample)

df_augmented = pd.DataFrame(new_samples)
df_balanced = pd.concat([df_balanced, df_augmented]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Final dataset size after augmentation: {len(df_balanced)}")

# **8️⃣ Feature Extraction with TF-IDF**
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_balanced['cleaned_text']).toarray()
y = df_balanced['label'].values

# **9️⃣ Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# **🔟 XGBoost Model**
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6
)
xgb_model.fit(X_train, y_train)

# Evaluate Model
train_accuracy = xgb_model.score(X_train, y_train)
test_accuracy = xgb_model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Cross-Validation
scores = cross_val_score(xgb_model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV accuracy: {scores.mean() * 100:.2f}%")

# **📊 Plot Feature Importance**
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.title('Top 10 Feature Importance (XGBoost)')
plt.show()

# **📊 Confusion Matrix & Classification Report**
y_pred = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# **11️⃣ Predict New URLs (Optional - Using your original URLs)**
new_urls = [
    "http://watshap-join.pubgproductions.com/login.php",
    "https://www.google.com",
    "http://condescending-hamilton-b3e79b.netlify.app/",
    "https://www.amazon.com",
    "https://kind-mayer-de4be4.netlify.app/"
]
cleaned_urls = [clean_text(url) for url in new_urls]
X_new = tfidf.transform(cleaned_urls).toarray()
predictions = xgb_model.predict(X_new)
label_map = {0: "Legitimate", 1: "Phishing"}
predicted_labels = [label_map[pred] for pred in predictions]

# Output Predictions
print("\nPredictions for new URLs:")
for url, label in zip(new_urls, predicted_labels):
    print(f"URL: {url} -> Predicted: {label}")