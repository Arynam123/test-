from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        # Vectorize the input message
        message_vectorized = vectorizer.transform([message])
        # Make prediction
        prediction = model.predict(message_vectorized)[0]
        return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
