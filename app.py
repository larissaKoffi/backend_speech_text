from flask import Flask, request, jsonify, make_response
from transformers import BartForConditionalGeneration, BartTokenizer
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Chargement du modèle
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def nettoyer(texte):
    texte = re.sub(r'\b(\w+)( \1\b)+', r'\1', texte) # mots doublés
    texte = re.sub(r'\b(euh+|heu+|hmm+|hum+)\b', '', texte, flags=re.IGNORECASE) # hésitations
    texte = re.sub(r'\s+', ' ', texte).strip() # espaces en trop
    return texte

def fn_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(
        inputs['input_ids'],max_length=50, min_length=5,num_beams=4, length_penalty=1.5, early_stopping=True, no_repeat_ngram_size=2
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# def _build_cors_preflight_response():
#     response = make_response()
#     response.headers.add("Access-Control-Allow-Origin", "*")
#     response.headers.add('Access-Control-Allow-Headers', "*")
#     response.headers.add('Access-Control-Allow-Methods', "*")
#     return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    texte = data.get('text', '')
    print("Texte reçu :", data)
    texte_propre = nettoyer(texte)
    summary = fn_summary(texte_propre)
    return _corsify_actual_response(jsonify({'summary': summary}))

if __name__ == "__main__":
    app.run(debug=True)
