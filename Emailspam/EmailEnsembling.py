import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import transformers  # Import the transformers module explicitly
from transformers import BertTokenizer, TFBertForSequenceClassification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import nlpaug.augmenter.word as naw  # For data augmentation

# Set seed for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Print TensorFlow and Transformers versions for debugging
print("TensorFlow version:", tf._version_)
print("Transformers version:", transformers._version_)

# Check if dataset exists
dataset_path = 'email_spam.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please provide the correct path.")

# Load dataset (Email Spam Dataset with word weights)
df = pd.read_csv(dataset_path, encoding='latin-1')

# Ensure the last column is the label
label_column = df.columns[-1]  # Assuming last column is the label (Prediction)
df.rename(columns={label_column: 'label'}, inplace=True)

# Convert label to binary (if not already)
df['label'] = LabelEncoder().fit_transform(df['label'])

# Convert word weight columns to numeric
df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Convert word weight columns into a single text representation
def convert_to_text(row):
    words = df.columns[:-1]  # Exclude label column
    text_representation = ' '.join([
        word for word, weight in zip(words, row[:-1]) if pd.to_numeric(weight, errors='coerce') > 0
    ])
    return text_representation

df['text'] = df.apply(convert_to_text, axis=1)

# Ensure text column has valid strings
df['text'] = df['text'].astype(str).fillna('')

# Balance dataset
ham = df[df['label'] == 0]
spam = df[df['label'] == 1]
target_count = max(len(ham), len(spam))

ham_downsampled = resample(ham, replace=False, n_samples=target_count, random_state=42)
spam_oversampled = resample(spam, replace=True, n_samples=target_count, random_state=42)
df_balanced = pd.concat([ham_downsampled, spam_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Data Augmentation
augmentor = naw.SynonymAug(aug_src='wordnet')

def augment_text(text):
    try:
        return augmentor.augment(text)[0]  # Take the first augmented result
    except:
        return text

# Ensure dataset reaches 10,000 samples
needed_samples = 10000 - len(df_balanced)
if needed_samples > 0:
    new_samples = df_balanced.sample(n=needed_samples, replace=True, random_state=42)
    new_samples['text'] = new_samples['text'].apply(augment_text)
    df_balanced = pd.concat([df_balanced, new_samples])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Final dataset size: {len(df_balanced)}")

# Check text column
print(df_balanced['text'].isna().sum())  # Count NaN values
print(df_balanced['text'].apply(lambda x: isinstance(x, str)).sum())  # Count string values
print(df_balanced['text'].head())  # Check the first few entries
df_balanced['text'] = df_balanced['text'].astype(str).fillna('')

# Define y
y = df_balanced['label'].values

# Tokenization with BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_encoded = tokenizer(df_balanced['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='tf')

# Verify shapes
print("X_encoded keys:", X_encoded.keys())
print("input_ids shape:", X_encoded['input_ids'].shape)
print("attention_mask shape:", X_encoded['attention_mask'].shape)
print("y shape:", y.shape)

# Get indices for train-test split
indices = np.arange(len(y))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

# Convert indices to TensorFlow tensors
train_indices = tf.convert_to_tensor(train_indices, dtype=tf.int32)
test_indices = tf.convert_to_tensor(test_indices, dtype=tf.int32)

# Construct X_train and X_test as dictionaries
X_train = {
    'input_ids': tf.gather(X_encoded['input_ids'], train_indices),
    'token_type_ids': tf.gather(X_encoded['token_type_ids'], train_indices),
    'attention_mask': tf.gather(X_encoded['attention_mask'], train_indices)
}
X_test = {
    'input_ids': tf.gather(X_encoded['input_ids'], test_indices),
    'token_type_ids': tf.gather(X_encoded['token_type_ids'], test_indices),
    'attention_mask': tf.gather(X_encoded['attention_mask'], test_indices)
}
y_train = y[train_indices.numpy()]
y_test = y[test_indices.numpy()]

# Verify split shapes
print("X_train['input_ids'] shape:", X_train['input_ids'].shape)
print("X_test['input_ids'] shape:", X_test['input_ids'].shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Define BERT Model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model with string optimizer identifier
model.compile(
    optimizer='adam',  # Use string identifier
    loss=loss_fn,
    metrics=['accuracy']
)

# Train model with a smaller batch size to avoid memory issues
history = model.fit(X_train, y_train, epochs=3, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot Training Results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Confusion Matrix & Classification Report
y_pred = np.argmax(model.predict(X_test, verbose=0).logits, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Optional: Save the model
model.save_pretrained('spam_classifier_model')
tokenizer.save_pretrained('spam_classifier_model')
print("Model and tokenizer saved to 'spam_classifier_model' directory.")import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import transformers  # Import the transformers module explicitly
from transformers import BertTokenizer, TFBertForSequenceClassification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import nlpaug.augmenter.word as naw  # For data augmentation

# Set seed for reproducibility
seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Print TensorFlow and Transformers versions for debugging
print("TensorFlow version:", tf._version_)
print("Transformers version:", transformers._version_)

# Check if dataset exists
dataset_path = 'email_spam.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please provide the correct path.")

# Load dataset (Email Spam Dataset with word weights)
df = pd.read_csv(dataset_path, encoding='latin-1')

# Ensure the last column is the label
label_column = df.columns[-1]  # Assuming last column is the label (Prediction)
df.rename(columns={label_column: 'label'}, inplace=True)

# Convert label to binary (if not already)
df['label'] = LabelEncoder().fit_transform(df['label'])

# Convert word weight columns to numeric
df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Convert word weight columns into a single text representation
def convert_to_text(row):
    words = df.columns[:-1]  # Exclude label column
    text_representation = ' '.join([
        word for word, weight in zip(words, row[:-1]) if pd.to_numeric(weight, errors='coerce') > 0
    ])
    return text_representation

df['text'] = df.apply(convert_to_text, axis=1)

# Ensure text column has valid strings
df['text'] = df['text'].astype(str).fillna('')

# Balance dataset
ham = df[df['label'] == 0]
spam = df[df['label'] == 1]
target_count = max(len(ham), len(spam))

ham_downsampled = resample(ham, replace=False, n_samples=target_count, random_state=42)
spam_oversampled = resample(spam, replace=True, n_samples=target_count, random_state=42)
df_balanced = pd.concat([ham_downsampled, spam_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Data Augmentation
augmentor = naw.SynonymAug(aug_src='wordnet')

def augment_text(text):
    try:
        return augmentor.augment(text)[0]  # Take the first augmented result
    except:
        return text

# Ensure dataset reaches 10,000 samples
needed_samples = 10000 - len(df_balanced)
if needed_samples > 0:
    new_samples = df_balanced.sample(n=needed_samples, replace=True, random_state=42)
    new_samples['text'] = new_samples['text'].apply(augment_text)
    df_balanced = pd.concat([df_balanced, new_samples])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Final dataset size: {len(df_balanced)}")

# Check text column
print(df_balanced['text'].isna().sum())  # Count NaN values
print(df_balanced['text'].apply(lambda x: isinstance(x, str)).sum())  # Count string values
print(df_balanced['text'].head())  # Check the first few entries
df_balanced['text'] = df_balanced['text'].astype(str).fillna('')

# Define y
y = df_balanced['label'].values

# Tokenization with BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_encoded = tokenizer(df_balanced['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='tf')

# Verify shapes
print("X_encoded keys:", X_encoded.keys())
print("input_ids shape:", X_encoded['input_ids'].shape)
print("attention_mask shape:", X_encoded['attention_mask'].shape)
print("y shape:", y.shape)

# Get indices for train-test split
indices = np.arange(len(y))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

# Convert indices to TensorFlow tensors
train_indices = tf.convert_to_tensor(train_indices, dtype=tf.int32)
test_indices = tf.convert_to_tensor(test_indices, dtype=tf.int32)

# Construct X_train and X_test as dictionaries
X_train = {
    'input_ids': tf.gather(X_encoded['input_ids'], train_indices),
    'token_type_ids': tf.gather(X_encoded['token_type_ids'], train_indices),
    'attention_mask': tf.gather(X_encoded['attention_mask'], train_indices)
}
X_test = {
    'input_ids': tf.gather(X_encoded['input_ids'], test_indices),
    'token_type_ids': tf.gather(X_encoded['token_type_ids'], test_indices),
    'attention_mask': tf.gather(X_encoded['attention_mask'], test_indices)
}
y_train = y[train_indices.numpy()]
y_test = y[test_indices.numpy()]

# Verify split shapes
print("X_train['input_ids'] shape:", X_train['input_ids'].shape)
print("X_test['input_ids'] shape:", X_test['input_ids'].shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Define BERT Model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model with string optimizer identifier
model.compile(
    optimizer='adam',  # Use string identifier
    loss=loss_fn,
    metrics=['accuracy']
)

# Train model with a smaller batch size to avoid memory issues
history = model.fit(X_train, y_train, epochs=3, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot Training Results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Confusion Matrix & Classification Report
y_pred = np.argmax(model.predict(X_test, verbose=0).logits, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Optional: Save the model
model.save_pretrained('spam_classifier_model')
tokenizer.save_pretrained('spam_classifier_model')
print("Model and tokenizer saved to 'spam_classifier_model' directory.")
