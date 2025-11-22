from datasets import load_dataset

# Load dataset
dataset = load_dataset("dmitva/human_ai_generated_text")
train_data = dataset['train']

# Prepare combined text data and labels
texts = []
labels = []

for sample in train_data:
    # Append human-written text with label 0
    texts.append(sample['human_text'])
    labels.append(0)
    
    # Append AI-generated text with label 1
    texts.append(sample['ai_text'])
    labels.append(1)

# Quick check on data size and example
print(f"Total samples: {len(texts)}")
print("Example human text:", texts[0])
print("Example AI-generated text:", texts[1])
print("Labels for first two samples:", labels[:2])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming texts and labels are from Step 1

# 1. Basic text preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters (keep only letters and digits)
    import re
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
texts_cleaned = [preprocess_text(txt) for txt in texts]

# 2. Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    texts_cleaned, labels, test_size=0.2, random_state=42, stratify=labels)

# 3. Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print("Training samples:", X_train_vec.shape)
print("Validation samples:", X_val_vec.shape)

