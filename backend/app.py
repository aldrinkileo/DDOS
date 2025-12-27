from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from model import CNN, osr_decision

app = Flask(__name__)
CORS(app)

model = CNN(20)
model.load_state_dict(torch.load("../saved_model/cnn_model.pth"))
model.eval()

class_centers = [torch.zeros(64), torch.ones(64)]

@app.route("/predict", methods=["POST"])
def predict():
    data = np.array(request.json["features"])
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    _, features = model(x)
    label, distance = osr_decision(features[0], class_centers)

    return jsonify({
        "prediction": label,
        "distance": round(distance, 3)
    })

app.run(debug=True)
