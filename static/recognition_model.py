import torch
import base64
import numpy as np
from io import BytesIO
from torch import nn
import base64
import io
from PIL import Image


def start_model():
    class_names_path = "static/class_names.txt"
    torch_model_path = "static/pytorch_model.bin"

    global LABELS
    LABELS = open(class_names_path).read().splitlines()

    global model
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1152, 256),
        nn.ReLU(),
        nn.Linear(256, len(LABELS)),
    )
    state_dict = torch.load(torch_model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()


def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.

    with torch.no_grad():
        out = model(x)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    values, indices = torch.topk(probabilities, 5)

    return {LABELS[i]: v.item() for i, v in zip(indices, values)}

# interface = gr.Interface(predict, inputs='sketchpad', outputs='label', live=True)
# interface.launch(debug=True, share=True)
