import os
import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

from model_transformer import create_dataloader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, '..', 'models', 'bert_classifier')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'val_processed.csv')
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.json')

df_val = pd.read_csv(DATA_PATH)

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)

# Dataset and DataLoader
val_loader = create_dataloader(df_val, tokenizer, max_len=256, batch_size=16, label_map=label_map, shuffle=False)

# Evaluation
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Report
label_names = [inv_label_map[i] for i in sorted(inv_label_map)]
report = classification_report(all_labels, all_preds, target_names=label_names)
print("Classification Report:\n", report)

cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

# Print to console
print("Confusion Matrix:\n")
print(cm_df.to_string())
