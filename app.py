# app.py
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])
    return render_template('index.html', prediction=prediction[0], text=text)

if __name__ == '__main__':
    app.run(debug=True)
