from flask import Flask, request, jsonify , render_template_string
import json
import re
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess text (simple tokenization and lowercasing)
def preprocess_text(text):
    text = re.sub(r'\\W+', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

# Generate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

# Load the FAQs from the JSON file
with open('data/keelworks_info.json', 'r') as file:
    data = json.load(file)

faqs = data['questions_and_answers']

# Precompute embeddings for FAQ questions
faq_embeddings = [get_bert_embedding(preprocess_text(faq['question'])) for faq in faqs]
faq_embeddings = np.vstack(faq_embeddings)  # Ensure embeddings are in a 2D array

# Find the best matching answer
def get_best_answer(user_query, faqs, faq_embeddings):
    preprocessed_query = preprocess_text(user_query)
    query_embedding = get_bert_embedding(preprocessed_query).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, faq_embeddings)
    best_match_index = similarities.argmax()

    return faqs[best_match_index]['answer']

# Endpoint for chatbot home
@app.route('/', methods=['GET'])
def home():
    html_form = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>KeelWorks Chatbot</title>
    </head>
    <body>
        <h1>Welcome to the KeelWorks Chatbot</h1>
        <form action="/ask" method="post">
            <label for="query">Enter your question:</label><br>
            <input type="text" id="query" name="query" required><br><br>
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    """
    return render_template_string(html_form)

# Endpoint for asking questions
@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form.get('query', '')
    answer = get_best_answer(user_query, faqs, faq_embeddings)
    result_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>KeelWorks Chatbot</title>
    </head>
     <body>
        <h1>KeelWorks Chatbot</h1>
        <form action="/ask" method="post">
            <label for="query">Enter your question:</label><br>
            <input type="text" id="query" name="query" required value="{{ query }}"><br><br>
            <input type="submit" value="Submit">
        </form>
        <h2>Answer:</h2>
        <p>{{ answer }}</p>
    </body>
    </html>
    """
    return render_template_string(result_template, query=user_query, answer=answer)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

