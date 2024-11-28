import torch
import torch.nn as nn
from model import Model

# AI 모델 정의

def load_model(model_path, IN, HIDDEN, OUT):
    model = Model(IN,HIDDEN,OUT)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, input_data):
    with torch.no_grad():
        input_tensor = torch.tensor([input_data], dtype=torch.float32)
        output = model(input_tensor).item()
    return output

