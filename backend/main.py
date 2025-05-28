import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Load intents.json
with open('intents.json', 'r') as file:
    data = json.load(file)

# Tokenizer
def tokenize(text):
    return text.lower().split()

# Prepare data
patterns, tags = [], []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(tokenize(pattern))
        tags.append(intent['tag'])

# Label encode tags
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)

# Build vocab
all_tokens = [token for pattern in patterns for token in pattern]
vocab = {"<pad>": 0, "<unk>": 1}
for token in set(all_tokens):
    vocab[token] = len(vocab)

# Encode text
encoded_patterns = [
    torch.tensor([vocab.get(token, vocab["<unk>"]) for token in pattern])
    for pattern in patterns
]

# Dataset
class IntentDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"]).long()
    return texts_padded, torch.tensor(labels).long()


# DataLoader
train_x, val_x, train_y, val_y = train_test_split(encoded_patterns, encoded_tags, test_size=0.2, random_state=42)
train_loader = DataLoader(IntentDataset(train_x, train_y), batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(IntentDataset(val_x, val_y), batch_size=16, shuffle=False, collate_fn=collate_fn)

# Model
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntentClassifier(len(vocab), 64, 128, len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, 201):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Accuracy = {correct / total:.4f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'label_encoder': label_encoder.classes_
}, 'intent_model.pth')

print("✅ Model saved to intent_model.pth")
