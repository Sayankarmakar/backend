from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Assuming nltk data is already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load the model
model_path = "multinomial_logistic_regression_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)
model.eval()
# Load the CountVectorizer used for preprocessing
vectorizer_path = "vectorizer.pkl"
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

# Define the genres
genres = ['business', 'world', 'sports', 'sci/tech']

# Preprocessing functions from the notebook
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    # Apply Porter stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    document_text = data['text']
    
    # Preprocess the input text
    processed_text = preprocess_text(document_text)
    inputs = vectorizer.transform([processed_text])
    inputs = torch.tensor(inputs.toarray(), dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    genre = genres[predicted.item()]
    
    return jsonify({'prediction': genre})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
