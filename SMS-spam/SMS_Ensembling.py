import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nlpaug.augmenter.word as naw
from wordcloud import WordCloud
import socket
import requests
import smtplib
from email.message import EmailMessage

# **1️⃣ Set Seed for Reproducibility**
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# **2️⃣ Download Stopwords**
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# **3️⃣ Load Dataset**
df = pd.read_csv("sms_spam.csv", encoding="latin-1")[["v1", "v2"]]
print(len(df))
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# **4️⃣ Text Cleaning Function**
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df["cleaned_text"] = df["text"].apply(clean_text)

# **5️⃣ Balance Dataset (Oversample if Needed)**
ham_count, spam_count = df["label"].value_counts()
df_ham = df[df["label"] == 0]
df_spam = df[df["label"] == 1].sample(n=ham_count, replace=True, random_state=42) if spam_count < ham_count else df[df["label"] == 1]

df_balanced = pd.concat([df_ham, df_spam]).sample(frac=1, random_state=42).reset_index(drop=True)

# **6️⃣ Data Augmentation (Expand to 10,000 Samples)**
augmentor = naw.SynonymAug(aug_src="wordnet")

def augment_text(text):
    try:
        return augmentor.augment(text)
    except:
        return text

new_samples = []
while len(df_balanced) + len(new_samples) < 10000:
    sample = df_balanced.sample(1).iloc[0].copy()
    sample["cleaned_text"] = augment_text(sample["cleaned_text"])
    new_samples.append(sample)

df_augmented = pd.DataFrame(new_samples)
df_balanced = pd.concat([df_balanced, df_augmented]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final dataset size: {len(df_balanced)} (Ham: {len(df_balanced[df_balanced['label'] == 0])}, Spam: {len(df_balanced[df_balanced['label'] == 1])})")

# **7️⃣ Tokenization & Padding**
max_vocab, max_length = 10000, 50
tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(df_balanced["cleaned_text"])

X = pad_sequences(tokenizer.texts_to_sequences(df_balanced["cleaned_text"]), maxlen=max_length)
y = df_balanced["label"].values

# **8️⃣ Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **9️⃣ Define CNN Model**
cnn_input = Input(shape=(max_length,))
cnn_embed = Embedding(input_dim=max_vocab, output_dim=128, input_length=max_length)(cnn_input)
cnn_conv = Conv1D(filters=128, kernel_size=5, activation="relu")(cnn_embed)
cnn_bn = BatchNormalization()(cnn_conv)
cnn_pool = GlobalMaxPooling1D()(cnn_bn)
cnn_dense = Dense(128, activation="relu")(cnn_pool)
cnn_dropout = Dropout(0.5)(cnn_dense)
cnn_output = Dense(1, activation="sigmoid")(cnn_dropout)
cnn_model = Model(inputs=cnn_input, outputs=cnn_output)

# **🔟 Define LSTM Model**
lstm_input = Input(shape=(max_length,))
lstm_embed = Embedding(input_dim=max_vocab, output_dim=128, input_length=max_length)(lstm_input)
lstm_layer = LSTM(128, return_sequences=False)(lstm_embed)
lstm_dropout = Dropout(0.2)(lstm_layer)
lstm_output = Dense(1, activation="sigmoid")(lstm_dropout)
lstm_model = Model(inputs=lstm_input, outputs=lstm_output)

# **1️⃣1️⃣ Merge CNN & LSTM Outputs**
merged = Concatenate()([cnn_output, lstm_output])
final_output = Dense(1, activation="sigmoid")(merged)
ensemble_model = Model(inputs=[cnn_input, lstm_input], outputs=final_output)

# **1️⃣2️⃣ Compile Model**
ensemble_model.compile(optimizer=Adam(learning_rate=0.0008), loss="binary_crossentropy", metrics=["accuracy"])

# **1️⃣3️⃣ Train Ensemble Model**
history = ensemble_model.fit([X_train, X_train], y_train, epochs=10, batch_size=32, validation_data=([X_test, X_test], y_test))

# **1️⃣4️⃣ Evaluate Model**
y_pred = (ensemble_model.predict([X_test, X_test]) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# **1️⃣5️⃣ Plot Training Results**
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Model Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Model Loss")
plt.show()

# # **1️⃣6️⃣ Confusion Matrix**
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(7, 7))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
# plt.xlabel("Predicted"), plt.ylabel("Actual"), plt.title("Confusion Matrix")
# plt.show()



# Predict on the entire dataset
y_full_pred = (ensemble_model.predict([X, X]) > 0.5).astype("int32")

# Generate confusion matrix
cm_full = confusion_matrix(y, y_full_pred)

# Plot confusion matrix
plt.figure(figsize=(7, 7))
sns.heatmap(cm_full, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Entire Dataset")
plt.show()








# def get_ip_from_url(url):
#     try:
#         domain = re.findall(r'://([^/]+)', url)[0]  # Extract domain
#         ip_address = socket.gethostbyname(domain)
#         return ip_address
#     except:
#         return None
# def get_location(ip_address):
#     response = requests.get(f"http://ip-api.com/json/{ip_address}")
#     return response.json()
# def send_alert_email(scammer_info):
#     msg = EmailMessage()
#     msg["Subject"] = "Spam Alert: Scammer Detected"
#     msg["From"] = "your_email@example.com"
#     msg["To"] = "authorities@example.com"
#     msg.set_content(f"Spam detected from {scammer_info['query']}\nLocation: {scammer_info['city']}, {scammer_info['country']}")

#     with smtplib.SMTP("smtp.gmail.com", 587) as server:
#         server.starttls()
#         server.login("your_email@example.com", "your_password")
#         server.send_message(msg)
# # Get IP and location
# spam_url = "http://example.com/scam"  # Replace with extracted URL
# ip = get_ip_from_url(spam_url)
# if ip:
#     scammer_info = get_location(ip)
#     send_alert_email(scammer_info)


# #Testing with new input
# input_data = {"phone": "+91XXXXXXXXXX", "text": "Congratulations! You won a free prize. Click the link now!, http://example.com/scam" }
# def preprocess_input_text(text):
#     text = clean_text(text)  # Use the same clean_text function
#     sequence = tokenizer.texts_to_sequences([text])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length)
#     return padded_sequence
# sms_input = preprocess_input_text(input_data["text"])
# prediction = (ensemble_model.predict([sms_input, sms_input]) > 0.5).astype("int32")

# if prediction[0][0] == 1:
#     print("Spam detected!")
#     # Extract URL
#     url = re.findall(r'(https?://\S+)', input_data["text"])
#     if url:
#         ip = get_ip_from_url(url[0])
#         if ip:
#             scammer_info = get_location(ip)
#             send_alert_email(scammer_info)
#         else:
#             print("IP address not found.")
#     else:
#         print("URL not found.")
# else:
#     print("Ham detected. The link is safe to click")


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get the predicted probabilities instead of class labels
y_proba = ensemble_model.predict([X_test, X_test])

# Compute False Positive Rate, True Positive Rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

