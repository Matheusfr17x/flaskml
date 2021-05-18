import numpy as np
from flask import Flask, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
names = pickle.load(open("iris_names.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)

    output = names[pred[0]]

    return render_template("index.html", prediction_text=output)
