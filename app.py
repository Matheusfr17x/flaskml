import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)

    output = pred[0]

    output2 = pred[0]/12
    
    text = str(output) 

    text = text[:-12]

    text2 = str(output2) 
    text2 = text2[:-12]
    return render_template("index.html", prediction_text="Valor Ganho Anunalmente U$ : " + text + " Ganhos mensais U$ : "+text2)


