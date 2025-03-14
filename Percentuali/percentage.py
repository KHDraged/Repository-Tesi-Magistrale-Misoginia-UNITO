import pandas as pd

# Caricare il dataset
file_path = "ricerca2_completa.csv"  # Modifica con il percorso corretto
df = pd.read_csv(file_path, sep=";")

# 1. Percentuale di 1 nella colonna misogyny_prediction
misogyny_percentage = (df["misogyny_prediction"].value_counts(normalize=True).get(1, 0)) * 100

# 2. Percentuali di 1 e 0 in irony_prediction e hatespeech_prediction tra i misogini
misogynistic_comments = df[df["misogyny_prediction"] == 1]
irony_counts = misogynistic_comments["irony_prediction"].value_counts(normalize=True) * 100
hate_speech_counts = misogynistic_comments["hatespeech_prediction"].value_counts(normalize=True) * 100

irony_percentage_1 = irony_counts.get(1, 0)
irony_percentage_0 = irony_counts.get(0, 0)
hate_speech_percentage_1 = hate_speech_counts.get(1, 0)
hate_speech_percentage_0 = hate_speech_counts.get(0, 0)

# 3. Totale e emozione più presente
emotion_columns = ["Anger", "Disgust", "Fear", "Joy", "Sadness"]
emotion_totals = df[emotion_columns].sum()
most_present_emotion = emotion_totals.idxmax()

# Stampare i risultati
print(f"Percentuale di 1 in misogyny_prediction: {misogyny_percentage:.2f}%")
print(f"Percentuale di 1 in irony_prediction tra misogini: {irony_percentage_1:.2f}%")
print(f"Percentuale di 0 in irony_prediction tra misogini: {irony_percentage_0:.2f}%")
print(f"Percentuale di 1 in hatespeech_prediction tra misogini: {hate_speech_percentage_1:.2f}%")
print(f"Percentuale di 0 in hatespeech_prediction tra misogini: {hate_speech_percentage_0:.2f}%")
print("Totali per ogni emozione:")
print(emotion_totals)
print(f"Emozione più presente: {most_present_emotion}")
