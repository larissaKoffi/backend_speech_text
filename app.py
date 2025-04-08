from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer
import re

app = Flask(__name__)

# Chargement du modèle
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def nettoyer(texte):
    texte = re.sub(r'\b(\w+)( \1\b)+', r'\1', texte) # mots doublés
    texte = re.sub(r'\b(euh+|heu+|hmm+|hum+)\b', '', texte, flags=re.IGNORECASE) # hésitations
    texte = re.sub(r'\s+', ' ', texte).strip() # espaces en trop
    return texte

def summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    texte = data.get('text', '')
    texte_propre = nettoyer(texte)
    summary = summary(texte_propre)
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
