import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import gradio as gr

# Function to load IMDb dataset
def load_imdb_data():
    base_path = os.path.abspath("C:/Users/Iran Computer/Desktop/aclImdb")
    texts = []
    labels = []

    for label in ['pos', 'neg']:
        folder_path = os.path.join(base_path, 'train', label)
        print(f"Listing files in {folder_path}:")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(1 if label == 'pos' else 0)

    return pd.DataFrame({'text': texts, 'label': labels})

data = load_imdb_data()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'], data['label'].reset_index(drop=True), test_size=0.2
)

# Use DistilBERT for faster training
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,            # Reduced epochs
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,
    warmup_steps=100,               # Reduced warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,               # Increased logging interval
    eval_steps=500,                 # Evaluate every 500 steps
    save_steps=500,                 # Save checkpoint every 500 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Create a sentiment analysis function
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    confidence = result['score']
    return f"Sentiment: {label}, Confidence: {confidence:.2f}"

# Build the Gradio interface
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis with DistilBERT",
    description="Enter a movie review to see if it's positive or negative."
)

interface.launch()