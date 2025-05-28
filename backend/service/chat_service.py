# service.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import json
import random

# Load model components
checkpoint = torch.load("intent_model.pth", map_location=torch.device("cpu"), weights_only=False)
vocab = checkpoint['vocab']
label_encoder = checkpoint['label_encoder']

# Define model
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# Load model
model = IntentClassifier(len(vocab), 64, 128, len(label_encoder))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Tokenize
def tokenize(text):
    return text.lower().split()

# Predict intent
def predict_intent(text):
    tokens = tokenize(text)
    input_tensor = torch.tensor([vocab.get(token, vocab["<unk>"]) for token in tokens])
    if len(input_tensor) == 0:
        return None
    input_tensor = pad_sequence([input_tensor], batch_first=True, padding_value=vocab["<pad>"]).long()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        return label_encoder[predicted_index]

# Get a random response for the predicted intent
def get_response(intent_label):
    for intent in intents['intents']:
        if intent['tag'] == intent_label:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."
