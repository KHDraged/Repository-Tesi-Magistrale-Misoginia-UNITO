import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm.auto import tqdm  # Usa tqdm.auto per compatibilità con Jupyter Notebook

# Carica il modello addestrato e il tokenizer
model_path = "final_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Carica il file CSV da analizzare
test_file = "ricerca3_resume.csv"  # Sostituisci con il nome del tuo file di test
test_data = pd.read_csv(test_file, sep=",", encoding="ISO-8859-1")

# Prepara i dati per l'inferenza con tqdm nella tokenizzazione
def preprocess_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        list(tqdm(texts, desc="Tokenizzazione")),
        padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return encodings

encodings = preprocess_texts(test_data["comment_text"], tokenizer)

input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

# Effettua le predizioni con tqdm nella fase di inferenza
batch_size = 32  # Dimensione del batch
predictions = []

with torch.no_grad():
    for i in tqdm(range(0, len(input_ids), batch_size), desc="Inferenza"):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]

        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        batch_predictions = torch.sigmoid(outputs.logits).cpu().numpy()
        predictions.append(batch_predictions)

# Ricomponi le predizioni in un array completo
predictions = np.vstack(predictions)

# Converti le probabilità in etichette binarie (1 se >= 0.5, altrimenti 0)
binary_predictions = (predictions >= 0.5).astype(int)

# Creiamo un DataFrame con le etichette binarie
binary_pred_df = pd.DataFrame(binary_predictions, columns=["Anger", "Disgust", "Fear", "Joy", "Sadness"])

# Unisci al file originale e salva
result_binary = pd.concat([test_data, binary_pred_df], axis=1)
result_binary.to_csv("ricerca3_st.csv", sep=";", index=False, encoding="utf-8")

print("File con etichette binarie salvato come 'ricerca2_st.csv'")

