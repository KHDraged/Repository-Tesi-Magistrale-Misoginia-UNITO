import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
import numpy as np

# Carica il tokenizer e il modello
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-italian-uncased", num_labels=2)

# Carica i dataset
train_file = 'training_ironita2018_anon_REV_.csv'


train_data = pd.read_csv(train_file, sep=";")


# Controlla i nomi delle colonne
print(train_data.columns)

# Crea il dataset PyTorch
class IronyDataset(Dataset):
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
train_dataset = IronyDataset(train_data['text'].values, train_data['irony'].values, tokenizer)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Funzione di training con Early Stopping
def train_model(model, train_dataloader, optimizer, patience=2, max_epochs=10):
    model.train()
    best_loss = np.inf
    patience_counter = 0

    for epoch in range(max_epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=True)
        total_loss = 0

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model.save_pretrained("best_model")  # Salva il miglior modello
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping attivato.")
            break

# Ottimizzatore
optimizer = AdamW(model.parameters(), lr=1e-5)

# Addestramento con Early Stopping
train_model(model, train_dataloader, optimizer, patience=2, max_epochs=3)

# Salvataggio finale
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
