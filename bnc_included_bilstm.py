import random
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import treebank
import os
from pathlib import Path
import lxml.etree as ET
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

########################################
# Load BNC Data
########################################

print("Loading BNC data...")

xml_files = Path('BNC/Texts').rglob('*.xml')
all_sentences = []  # List of all sentences
all_tags = set()
all_words = set()

# Load all BNC sentences first
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find all sentences (BNC uses <s> tag for sentences)
    for sentence in root.findall('.//s'):
        word_tag_pairs = []
        
        # Extract words within this sentence
        for word in sentence.findall('.//w'):
            word_text = word.text
            pos_tag = word.get('c5')
            
            if word_text and pos_tag:
                word_tag_pairs.append((word_text, pos_tag))
                all_words.add(word_text)
                all_tags.add(pos_tag)
        
        # Only add non-empty sentences
        if word_tag_pairs:
            all_sentences.append(word_tag_pairs)
random.seed(42)
random.shuffle(all_sentences)
pct = 5  # percentage of sentences to use              
sentences_A = random.sample(all_sentences,k=max(1, len(all_sentences) * pct // 100))

print(f"Using {len(sentences_A):,} sentences out of {len(all_sentences):,} total sentences in BNC ({len(sentences_A)/len(all_sentences):.2%})")

# Check what you got
print(f"Total sentences: {len(sentences_A):,}")
print(f"Total unique words: {len(all_words):,}")
print(f"Total unique tags: {len(all_tags)}")

########################################
# Load Tagged Testdata from Liguistik HIWI
########################################

print("\nLoading test sentences from external file...")

test_sentences = []
test_path = "POS-Tagger/POS-Tagging-Testdaten/Digital Chat Daten Annotiert.txt"
test_path_2 = "POS-Tagger/POS-Tagging-Testdaten/Essay 2.txt"
test_path_3 = "POS-Tagger/POS-Tagging-Testdaten/CLAWS Verbessert Neu.txt"
test_path_4 = "POS-Tagger/POS-Tagging-Testdaten/Picture Description.txt"

with open(test_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)

with open(test_path_2, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)

with open(test_path_3, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)

with open(test_path_4, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)

# split test_sentences into test and train sentences randomly
#random.shuffle(test_sentences)
#split_index = int(0.5 * len(test_sentences))
#train_sentences = test_sentences[:split_index]
#test_sentences = test_sentences[split_index:]

print(f"Loaded {len(test_sentences):,} sentences from external test data.")
#print(f"Loaded {len(train_sentences):,} sentences from external train data.")
# print all tags in test sentences
test_tags = set(tag for sent in test_sentences for _, tag in sent)
print(f"Total unique tags in test sentences: {len(test_tags)}")

########################################
# Setup & Data Loading
########################################
# see if cuda or mps is available, otherwise use cpu
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

all_sentences = sentences_A
#all_sentences.extend(train_sentences)  # Add external train sentences to the pool for splitting

# split for train, val, test
split1 = int(0.9 * len(all_sentences))
split2 = int(0.95 * len(all_sentences)) 
train_data = all_sentences[:split1]
val_data = all_sentences[split1:split2]
test_data = all_sentences[split2:]


print(f"Training sentences: {len(train_data):,}")
print(f"Validation sentences: {len(val_data):,}")
print(f"Testing sentences: {len(test_data):,}")
# print all tags in training and in testing
train_tags = set(tag for sent in train_data for _, tag in sent)
val_tags = set(tag for sent in val_data for _, tag in sent)
test_tags = set(tag for sent in test_data for _, tag in sent)
print(f"Unique tags in training: {len(train_tags):,}")
print(f"Unique tags in validation: {len(val_tags):,}")
print(f"Unique tags in testing: {len(test_tags):,}")

########################################
# Vocabulary Construction
########################################

word2idx = {"<PAD>": 0, "<UNK>": 1}
# reserve index 0 for unknown tags so we can map unseen labels to a class
tag2idx = {"<UNK>": 0}

for sent in train_data:
    for word, tag in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)

# Do not extend tag set with external test labels; unknown tags are handled
# by mapping to the <UNK> index above.

idx2tag = {v: k for k, v in tag2idx.items()}

########################################
# Encoding Utilities
########################################

def encode_sentence(sent):
    words = [word2idx.get(w, word2idx["<UNK>"]) for w, _ in sent]
    # unknown POS tags map to the <UNK> class instead of causing an error
    tags = [tag2idx.get(t, tag2idx["<UNK>"]) for _, t in sent]
    return words, tags

def pad_batch(batch):
    max_len = max(len(x[0]) for x in batch)
    X, Y = [], []
    for words, tags in batch:
        pad_len = max_len - len(words)
        X.append(words + [0] * pad_len)
        Y.append(tags + [-1] * pad_len)  # -1 will be ignored in loss
    return torch.tensor(X, device=device), torch.tensor(Y, device=device)

########################################
# Model Definition
########################################

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out)
        return logits
    
########################################
# Training Setup
########################################

print("\nSetting up model, optimizer, and loss function...")

model = BiLSTMTagger(len(word2idx), len(tag2idx)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=-1)


########################################
# Training Loop
########################################

def train(model, train_data, val_data, epochs=5, batch_size=32): 
    global_step = 0

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        total_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            encoded = [encode_sentence(s) for s in batch]
            X, Y = pad_batch(encoded)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits.view(-1, logits.shape[-1]), Y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ---- TensorBoard: batch loss
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)

            # ---- Validation every 1000 batches
            if global_step % 1000 == 0 and global_step > 0:
                validate(model, val_data, global_step)
                model.train()  # back to train mode after validation

            global_step += 1

        avg_loss = total_loss / (len(train_data) // batch_size)
        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f}")


def validate(model, data, step, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            encoded = [encode_sentence(s) for s in batch]
            X, Y = pad_batch(encoded)

            logits = model(X)
            loss = criterion(
                logits.view(-1, logits.shape[-1]),
                Y.view(-1)
            )
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            mask = Y != -1
            correct += (preds[mask] == Y[mask]).sum().item()
            total += mask.sum().item()

    acc = correct / total
    avg_loss = total_loss / max(1, len(data) // batch_size)

    print(f"Validation | loss: {avg_loss:.4f}, acc: {acc:.4f}")
    writer.add_scalar("Loss/validation", avg_loss, step)
    writer.add_scalar("Accuracy/validation", acc, step)


########################################
# Evaluation
########################################

def evaluate(model, data):
    
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for sent in data:
            words, gold_tags = encode_sentence(sent)
            X = torch.tensor([words]).to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=-1)[0]

            for p, g in zip(preds, gold_tags):
                total += 1
                correct += (p.item() == g)
                all_preds.append(p.item())
                all_golds.append(g)

    acc = correct / total
    f1 = f1_score(all_golds, all_preds, average='weighted', zero_division=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1 score: {f1:.4f}")


########################################
# Main Execution
########################################
writer = SummaryWriter(log_dir="runs/bnc_bilstm_pos_exploratory_partition_from_all")

print("Training model...")
train(model, train_data, val_data, epochs=5)

print("\nEvaluating model...")
evaluate(model, test_data)
writer.close()


########################################
# Test with own testdata
########################################

# F1-Score
# Accuracy
print("\nEvaluating model on external test sentences...")
evaluate(model, test_sentences)
writer.close()

print("\nDone.")