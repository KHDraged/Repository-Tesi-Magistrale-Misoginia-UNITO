import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm  # Importa tqdm per la barra di progresso

# 🟢 1. Imposta il device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Stai usando:", device)

# 🟢 2. Carica il modello addestrato
model_path = "final_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()  # Mettiamo il modello in modalità valutazione

# 🟢 3. Carica il dataset di commenti
input_file = "ricerca3_resume.csv"
df = pd.read_csv(input_file, sep=",", encoding="ISO-8859-1")

# Assumiamo che la colonna dei commenti si chiami "comment_text"
if "comment_text" not in df.columns:
    raise KeyError("La colonna 'comment_text' non è presente nel dataset!")

# 🟢 4. Definiamo un dataset per PyTorch
class CommentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

# 🟢 5. Creiamo il DataLoader
dataset = CommentDataset(df["comment_text"].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=8)

# 🟢 6. Classifichiamo i commenti con una barra di progresso
predictions = []

# Usa tqdm per visualizzare la barra di progresso
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Classificando commenti", ncols=100):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())

# 🟢 7. Salviamo i risultati
df["hatespeech_prediction"] = predictions
output_file = "ricerca3_hs.csv"
df.to_csv(output_file, index=False)

print(f"Classificazione completata! Risultati salvati in {output_file}")
