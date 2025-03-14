import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from tqdm import tqdm
import numpy as np

# Carica il tokenizer e il modello
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-italian-uncased", num_labels=2)

# Carica i dataset
train_file_raw = 'AMI2020_training_raw_anon.tsv'
train_file_synt = 'AMI2020_training_synt.tsv'

train_data_raw = pd.read_csv(train_file_raw, sep="\t")
train_data_synt = pd.read_csv(train_file_synt, sep="\t")

# Crea il dataset PyTorch
class MisogynyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Creazione dataset
train_dataset_raw = MisogynyDataset(train_data_raw['text'].values, train_data_raw['misogynous'].values, tokenizer)
train_dataset_synt = MisogynyDataset(train_data_synt['text'].values, train_data_synt['misogynous'].values, tokenizer)

# DataLoader
train_dataloader = DataLoader(train_dataset_raw, batch_size=8, shuffle=True)

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-italian-uncased", num_labels=2)
model.to(device)

# Funzione di training con Early Stopping
def train_model(model, train_dataloader, optimizer, patience=2, max_epochs=10):
    model.train()
    best_loss = np.inf
    patience_counter = 0
    
    for epoch in range(max_epochs):
        total_loss = 0
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
        
        # Salvataggio del modello ad ogni epoca
        model.save_pretrained(f"model_epoch_{epoch+1}")
        tokenizer.save_pretrained(f"model_epoch_{epoch+1}")

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Imposta l'ottimizzatore
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Corsa del training
train_model(model, train_dataloader, optimizer, max_epochs=3)

# Salvataggio finale
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")

print("Training completato e modello salvato.")
