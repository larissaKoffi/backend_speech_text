from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
import torchaudio
import numpy as np
import io, time
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"], supports_credentials=True)


#dictionnaire pour mettre les emotions en francais

EMOTION_LABELS = {
    "angry": "colère",
    "disgust": "dégoût",
    "fear": "peur",
    "happy": "joie",
    "neutral": "neutre",
    "sad": "tristesse",
    "surprise": "surprise"
}

# Résumé de texte
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-abstract")
model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-abstract").to(device)

# Modèle de détection émotion vocale
emotion_pipeline = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# Nettoyage texte
import re
def nettoyer(texte):
    texte = re.sub(r'\b(\w+)( \1\b)+', r'\1', texte)
    texte = re.sub(r'\b(euh+|heu+|hmm+|hum+)\b', '', texte, flags=re.IGNORECASE)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def split_text(text, max_tokens=1024):
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        temp_chunk = current_chunk + " " + sentence
        if len(tokenizer(temp_chunk)['input_ids']) < max_tokens:
            current_chunk = temp_chunk
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

#résumé de texte

def fn_summary(text):
    chunks = split_text(text)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
        input_len = inputs['input_ids'].shape[1]
        max_len = max(60, min(int(input_len * 0.7), 250))
        min_len = max(30, min(int(input_len * 0.3), max_len - 10))
        summary_ids = model.generate(
            inputs['input_ids'], max_length=max_len, min_length=min_len,
            num_beams=4, length_penalty=1.2, early_stopping=True,
            no_repeat_ngram_size=2
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return " ".join(summaries)



#detection de sentiment


@app.route('/analyze', methods=['POST', 'OPTIONS'])
@cross_origin(origins=["https://frontend-speech-text.vercel.app/", "http://127.0.0.1:5500"], supports_credentials=True)
def analyze():
    try:
        transcript = request.form['text']
        audio_file = request.files['audio']

        print("Texte reçu :", transcript)
        print("Nom fichier audio :", audio_file.filename)
        print("MIME Type :", audio_file.content_type)

        audio_segment = AudioSegment.from_file(audio_file, format="webm")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        waveform, sample_rate = torchaudio.load(wav_io)
        result = emotion_pipeline(waveform.squeeze().numpy())
        label = max(result, key=lambda x: x['score'])['label']
        emotion = EMOTION_LABELS.get(label, label)

        summary = fn_summary(nettoyer(transcript))

        print("Résumé :", summary)
        print("Émotion détectée :", emotion)

        return jsonify({'summary': summary, 'emotion': emotion})

    except Exception as e:
        import traceback
        print("Erreur backend :", str(e))
        traceback.print_exc()  # 🔥 pour voir la stack complète
        return jsonify({'error': 'Erreur de traitement audio'}), 500




if __name__ == "__main__":
    app.run(debug=True)
