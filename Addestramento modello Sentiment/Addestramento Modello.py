import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Carica il modello addestrato e il tokenizer
model_path = "final_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Carica il file CSV da analizzare
test_file = "resume3_st.csv"  # Sostituisci con il nome del tuo file di test
test_data = pd.read_csv(test_file, sep=",", encoding="ISO-8859-1")

# Prepara i dati per l'inferenza
def preprocess_texts(texts, tokenizer, max_length=512):
    encodings = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encodings

encodings = preprocess_texts(test_data["comment_text"], tokenizer)
input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

# Effettua le predizioni
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.sigmoid(outputs.logits)  # Per multi-label, usiamo la sigmoide
    predictions = predictions.cpu().numpy()

# Converti le probabilità in etichette binarie (1 se >= 0.5, altrimenti 0)
binary_predictions = (predictions >= 0.5).astype(int)

# Creiamo un DataFrame con le etichette binarie
binary_pred_df = pd.DataFrame(binary_predictions, columns=["Anger", "Disgust", "Fear", "Joy", "Sadness"])

# Unisci al file originale e salva
result_binary = pd.concat([test_data, binary_pred_df], axis=1)
result_binary.to_csv("resume3_st.csv", sep=";", index=False, encoding="utf-8")

print("File con etichette binarie salvato come 'resume3_st.csv'")

