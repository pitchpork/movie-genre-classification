import os
import random
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 256
MODEL_NAME = 'bert-base-uncased'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models', 'bert_classifier')

# Ensure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label_map):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = label_map

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize input text and convert label to tensor
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.label_map[label], dtype=torch.long)
        }


def create_label_map(genres):
    unique = sorted(set(genres))
    return {genre: idx for idx, genre in enumerate(unique)}


def create_dataloader(df, tokenizer, max_len, batch_size, label_map, shuffle=True):
    dataset = MovieDataset(
        texts=df['Description_clean'].tolist(),
        labels=df['Genre'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len,
        label_map=label_map
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    running_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
    return running_loss / len(dataloader)  # Average loss for the epoch


def evaluate_model(model, dataloader):
    model.eval()
    preds, true_labels = [], []
    eval_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            eval_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())  # Predicted labels
            true_labels.extend(labels.cpu().numpy())  # True labels

    avg_loss = eval_loss / len(dataloader)
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    return avg_loss, acc, f1, preds, true_labels


def main():
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_processed.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_processed.csv'))

    # Create and save label map
    label_map = create_label_map(train_df['Genre'])
    with open(os.path.join(MODEL_DIR, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)

    # Tokenizer and model initialization
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map))
    model.to(DEVICE)

    # DataLoaders
    train_loader = create_dataloader(train_df, tokenizer, MAX_LEN, BATCH_SIZE, label_map, shuffle=True)
    val_loader = create_dataloader(val_df, tokenizer, MAX_LEN, BATCH_SIZE, label_map, shuffle=False)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop with history tracking
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_f1, preds, true_labels = evaluate_model(model, val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)
        print(
            f'Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f}')

    # Save model, tokenizer, and metrics
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Generate and save classification report
    report = classification_report(true_labels, preds, target_names=list(label_map.keys()), output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(MODEL_DIR, 'classification_report.csv'), index=True)

    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, preds)
    cm_df = pd.DataFrame(cm, index=list(label_map.keys()), columns=list(label_map.keys()))
    cm_df.to_csv(os.path.join(MODEL_DIR, 'confusion_matrix.csv'))

    # Plot and save loss curves
    epochs = len(history['train_loss'])
    x = list(range(1, epochs + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(x, history['train_loss'], label='Train Loss', linewidth=2, marker='o')
    plt.plot(x, history['val_loss'], label='Validation Loss', linewidth=2, marker='s')

    plt.title('Training and Validation Loss Over Epochs', fontsize=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.xticks(x)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(MODEL_DIR, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Model, metrics, and plots saved to {MODEL_DIR}')


if __name__ == '__main__':
    main()
