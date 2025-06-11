import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import nlpaug.augmenter.word as naw  # Replacing textaugment with nlpaug

# **1️⃣ Set Seed for Reproducibility**
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# **2️⃣ Download Stopwords**
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# **3️⃣ Load the UCI SMS Spam Dataset**
df = pd.read_csv('sms_spam.csv', encoding='latin-1')  # Update path if needed
df.dropna(subset=['v2'], inplace=True)  # Drop missing values

# **4️⃣ Text Cleaning Function**
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_text'] = df['v2'].apply(clean_text)

# **5️⃣ Label Encoding**
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['v1'])  # Ham → 0, Spam → 1

# **6️⃣ Balance Data: Equal Ham & Spam Distribution**
df_ham = df[df['label'] == 0]  # Ham messages
df_spam = df[df['label'] == 1]  # Spam messages

# Find majority & minority class sizes
ham_count = len(df_ham)
spam_count = len(df_spam)
target_count = max(ham_count, spam_count)

# **Undersample Ham & Oversample Spam**
df_ham_downsampled = resample(df_ham, replace=False, n_samples=target_count, random_state=42)
df_spam_oversampled = resample(df_spam, replace=True, n_samples=target_count, random_state=42)

# Combine Balanced Data
df_balanced = pd.concat([df_ham_downsampled, df_spam_oversampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset

print(f"New dataset size: {len(df_balanced)}")
print(f"New ham count: {len(df_balanced[df_balanced['label'] == 0])}")
print(f"New spam count: {len(df_balanced[df_balanced['label'] == 1])}")

# **7️⃣ Data Augmentation (Increase Dataset to 10,000 Samples)**
augmentor = naw.SynonymAug(aug_src='wordnet')  # Use NLPaug for synonym replacement

def augment_text(text):
    try:
        return augmentor.augment(text)  # Replace words with synonyms
    except:
        return text  # If error, return original text

# Generate Additional Samples
new_samples = []
while len(df_balanced) + len(new_samples) < 10000:
    sample = df_balanced.sample(1).iloc[0].copy()
    sample['cleaned_text'] = augment_text(sample['cleaned_text'])
    new_samples.append(sample)

# Convert List to DataFrame & Add to Dataset
df_augmented = pd.DataFrame(new_samples)
df_balanced = pd.concat([df_balanced, df_augmented]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final dataset size after augmentation: {len(df_balanced)}")

# **8️⃣ Tokenization & Padding**
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_balanced['cleaned_text'])
X = tokenizer.texts_to_sequences(df_balanced['cleaned_text'])

max_sequence_length = 100  # Fixed input size for LSTM
X_pad = pad_sequences(X, maxlen=max_sequence_length)

y = df_balanced['label'].values  # Labels

# **9️⃣ Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# **🔟 Define LSTM Model**
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# **📊 Plot Training Results**
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# **📊 Confusion Matrix & Classification Report**
y_pred = (model.predict(X_test) > 0.5)  # Convert probabilities to binary

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
