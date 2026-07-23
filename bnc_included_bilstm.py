import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import treebank
import os
from pathlib import Path
from copy import deepcopy
import lxml.etree as ET
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

########################################
# Load BNC Data
########################################

print("Loading BNC data...")

xml_files = list(Path("Code/BNC/Texts").rglob("*.xml"))
pct = 5  # percentage of sentences to use

# First pass: count how many non-empty sentences exist in the BNC corpus.
# This lets us keep only a 5% sample in memory instead of loading everything up front.
total_sentence_count = 0
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for sentence in root.findall(".//s"):
        word_tag_pairs = []
        for word in sentence.findall(".//w"):
            word_text = word.text
            pos_tag = word.get("c5")
            if word_text and pos_tag:
                word_tag_pairs.append((word_text, pos_tag))

        if word_tag_pairs:
            total_sentence_count += 1

sample_size = max(1, total_sentence_count * pct // 100)
random.seed(42)
torch.manual_seed(42)

sampled_sentences = []
seen_sentences = 0

# Second pass: reservoir sample the sentences so only the selected subset is kept in memory.
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for sentence in root.findall(".//s"):
        word_tag_pairs = []
        for word in sentence.findall(".//w"):
            word_text = word.text
            pos_tag = word.get("c5")
            if word_text and pos_tag:
                word_tag_pairs.append((word_text, pos_tag))

        if not word_tag_pairs:
            continue

        seen_sentences += 1
        if len(sampled_sentences) < sample_size:
            sampled_sentences.append(word_tag_pairs)
        else:
            replace_idx = random.randrange(seen_sentences)
            if replace_idx < sample_size:
                sampled_sentences[replace_idx] = word_tag_pairs

sentences_A = sampled_sentences
all_words = {word for sent in sentences_A for word, _ in sent}
all_tags = {tag for sent in sentences_A for _, tag in sent}

print(
    f"Using {len(sentences_A):,} sentences out of {total_sentence_count:,} total sentences in BNC ({len(sentences_A)/max(1, total_sentence_count):.2%})"
)

# Check what you got
print(f"Total sentences: {len(sentences_A):,}")
print(f"Total unique words: {len(all_words):,}")
print(f"Total unique tags: {len(all_tags)}")

########################################
# Load Tagged Testdata and Split into Age Groups
########################################

test_sentences_klasse_5 = []
test_sentences_klasse_6 = []
test_sentences_klasse_7 = []
test_sentences_klasse_8 = []
test_sentences_klasse_9 = []
test_sentences_klasse_10 = []
test_sentences_klasse_11 = []
test_sentences_klasse_12 = []

def extract_age_group(sentence):
    """
    Extracts the age group from the metadata line of the test data.
    The age group is indicated by a pattern that consists of "Klasse" followed by a number.
    Args:
        sentence (str): The input sentence which may contain the age group information.
    Returns:
      the number as an integer if found, otherwise returns None.
    """
    age_pattern = re.compile(r"(\s*Klasse\d+)")
    match = age_pattern.search(sentence)
    if match:
        age = match.group(1).strip()
        # extract the number from the age string
        age_number = int(re.search(r"\d+", age).group())
        # assign to class based on age number and return correct test_sentences group
        return age_number
    return None


def store_sentence_by_age_group(sentence, age_group):
    """
    Stores the given sentence in the appropriate list based on the age group.
    Args:
        sentence (list of tuples): The sentence to be stored, where each tuple contains a word and its corresponding tag.
        age_group (int): The age group number extracted from the metadata.
    """
    match age_group:
        case 5:
            test_sentences_klasse_5.append(sentence)
        case 6:
            test_sentences_klasse_6.append(sentence)
        case 7:
            test_sentences_klasse_7.append(sentence)
        case 8:
            test_sentences_klasse_8.append(sentence)
        case 9:
            test_sentences_klasse_9.append(sentence)
        case 10:
            test_sentences_klasse_10.append(sentence)
        case 11:
            test_sentences_klasse_11.append(sentence)
        case 12:
            test_sentences_klasse_12.append(sentence)

########################################
# Load Tagged Testdata
########################################

print("\nLoading test sentences from external file...")

test_sentences = []
test_path = "POS-Tagger/POS-Tagging-Testdaten/Digital Chat Daten Annotiert.txt"
test_path_2 = "POS-Tagger/POS-Tagging-Testdaten/Essay 2.txt"
test_path_3 = "POS-Tagger/POS-Tagging-Testdaten/CLAWS Verbessert Neu.txt"
test_path_4 = "POS-Tagger/POS-Tagging-Testdaten/Picture Description.txt"
test_path_5 = "POS-Tagger/POS-Tagging-Testdaten/Short Essay.txt"
test_sentences_essay = []
test_sentences_chat = []
test_sentences_picture = []
test_sentences_short_essay = []

with open(test_path, "r", encoding="utf-8") as f:
    empty = True  # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue  # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            # ignore everything that the instructor says -> I_ZZ0
            if line.startswith("I_ZZ0"):
                continue
            if line.startswith("S_ZZ0"):
                tokens = tokens[
                    2:
                ]  # remove the first two tokens which is the metadata about who is speaking
            for token in tokens:
                if "_" in token and len(token.rsplit("_", 1)) == 2:
                    word, tag = token.rsplit("_", 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                test_sentences_chat.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True

with open(test_path_2, "r", encoding="utf-8") as f:
    empty = True  # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue  # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if "_" in token and len(token.rsplit("_", 1)) == 2:
                    word, tag = token.rsplit("_", 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                test_sentences_essay.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True

# Data that was used to test CLAWS etc.
with open(test_path_3, "r", encoding="utf-8") as f:
    empty = True  # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue  # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if "_" in token and len(token.rsplit("_", 1)) == 2:
                    word, tag = token.rsplit("_", 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True 

with open(test_path_4, "r", encoding="utf-8") as f:
    empty = True  # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue  # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if "_" in token and len(token.rsplit("_", 1)) == 2:
                    word, tag = token.rsplit("_", 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                test_sentences_picture.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True
with open(test_path_5, "r", encoding="utf-8") as f:
    empty = True  # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue  # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if "_" in token and len(token.rsplit("_", 1)) == 2:
                    word, tag = token.rsplit("_", 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                test_sentences_short_essay.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True

# split test_sentences into test and train sentences randomly
random.shuffle(test_sentences)
random.shuffle(test_sentences_essay)
random.shuffle(test_sentences_chat)
random.shuffle(test_sentences_picture)
random.shuffle(test_sentences_short_essay)
split_index_essay = int(0.8 * len(test_sentences_essay))
split_index_chat = int(0.8 * len(test_sentences_chat))
split_index_picture = int(0.8 * len(test_sentences_picture))
split_index_short_essay = int(0.8 * len(test_sentences_short_essay))

print(f"Loaded {len(test_sentences):,} sentences from own data.")
# print all tags in test sentences
test_tags = set(tag for sent in test_sentences for _, tag in sent)
print(f"Total unique tags in test sentences: {len(test_tags)}")

# split sentences into train (80%) and test (20%) by category
essay_train = test_sentences_essay[:split_index_essay]
chat_train = test_sentences_chat[:split_index_chat]
picture_train = test_sentences_picture[:split_index_picture]
short_essay_train = test_sentences_short_essay[:split_index_short_essay]

essay_test = test_sentences_essay[split_index_essay:]
chat_test = test_sentences_chat[split_index_chat:]
picture_test = test_sentences_picture[split_index_picture:]
short_essay_test = test_sentences_short_essay[split_index_short_essay:]

# remove all train data from test_sentences
test_sentences = [s for s in test_sentences if s not in essay_train and s not in chat_train and s not in picture_train and s not in short_essay_train]

print(f"Total test sentences after split: {len(test_sentences):,}")

# print test and train for each category
print(f"Train sentences for essays: {len(essay_train):,}, Test sentences for essays: {len(essay_test):,}")
print(f"Train sentences for chat: {len(chat_train):,}, Test sentences for chat: {len(chat_test):,}")
print(f"Train sentences for picture description: {len(picture_train):,}, Test sentences for picture description: {len(picture_test):,}")
print(f"Train sentences for short essays: {len(short_essay_train):,}, Test sentences for short essays: {len(short_essay_test):,}")

print(f"Total test sentences after split: {len(test_sentences):,}")

# print length of each age group test sentences
print(f"Test sentences for Klasse 5: {len(test_sentences_klasse_5):,}")
print(f"Test sentences for Klasse 6: {len(test_sentences_klasse_6):,}")
print(f"Test sentences for Klasse 7: {len(test_sentences_klasse_7):,}")
print(f"Test sentences for Klasse 8: {len(test_sentences_klasse_8):,}")
print(f"Test sentences for Klasse 9: {len(test_sentences_klasse_9):,}")
print(f"Test sentences for Klasse 10: {len(test_sentences_klasse_10):,}")
print(f"Test sentences for Klasse 11: {len(test_sentences_klasse_11):,}")
print(f"Test sentences for Klasse 12: {len(test_sentences_klasse_12):,}")

########################################
# Setup & Data Loading
########################################
# see if cuda or mps is available, otherwise use cpu
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

# split for train, val, test
split1 = int(0.9 * len(sentences_A))
split2 = int(0.95 * len(sentences_A))
train_data = sentences_A[:split1]
val_data = sentences_A[split1:split2]
test_data = sentences_A[split2:]

train_data += essay_train + chat_train + picture_train + short_essay_train

print(f"Train sentences: {len(train_data):,}")
print(f"Validation sentences: {len(val_data):,}")
print(f"Test sentences: {len(test_data):,}")

# shuffel train data after adding the test sentences to it
random.shuffle(train_data)

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

#BERT tokenizer and model TODO 
from transformers import BertTokenizerFast, BertModel

MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

tag2idx = {"<UNK>": 0}
for sent in train_data:
    for word, tag in sent:
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)
idx2tag = {v: k for k, v in tag2idx.items()}


#word2idx = {"<PAD>": 0, "<UNK>": 1}
# reserve index 0 for unknown tags so we can map unseen labels to a class
#tag2idx = {"<UNK>": 0}

#for sent in train_data:
#    for word, tag in sent:
#        if word not in word2idx:
#            word2idx[word] = len(word2idx)
#        if tag not in tag2idx:
#            tag2idx[tag] = len(tag2idx)

#idx2tag = {v: k for k, v in tag2idx.items()}

########################################
# Encoding Utilities
########################################


#def encode_sentence(sent):
#    """Encodes a sentence into word indices and tag indices.
#    Unknown words are mapped to the <UNK> index, and unknown tags are also mapped to the <UNK> index.#
#
#    Args:        
#        sent (list of tuples): A sentence represented as a list of (word, tag) tuples
#    Returns:        
#        words (list of int): List of word indices corresponding to the input sentence
#        tags (list of int): List of tag indices corresponding to the input sentence
#    """
#    words = [word2idx.get(w, word2idx["<UNK>"]) for w, _ in sent]
#    # unknown POS tags map to the <UNK> class instead of causing an error
#    tags = [tag2idx.get(t, tag2idx["<UNK>"]) for _, t in sent]
#    return words, tags


MAX_SEQ_LENGTH = 128


def encode_sentence(sent): # TODO
    words = [w for w, _ in sent]
    tags = [t for _, t in sent]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    word_ids = encoding.word_ids()

    label_ids = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-1)
        elif word_idx != prev_word_idx:
            label_ids.append(tag2idx.get(tags[word_idx], tag2idx["<UNK>"]))
        else:
            label_ids.append(-1)
        prev_word_idx = word_idx

    return encoding["input_ids"], encoding["attention_mask"], label_ids

def pad_batch(batch): # TODO
    lengths = [len(ids) for ids, _, _ in batch]
    max_len = max(lengths)
    if not hasattr(pad_batch, "_sequence_stats_printed"):
        print(
            f"[debug] sequence length stats: min={min(lengths)}, median={np.median(lengths):.1f}, max={max_len}"
        )
        pad_batch._sequence_stats_printed = True

    input_ids, attn_masks, Y = [], [], []
    for ids, mask, tags in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        attn_masks.append(mask + [0] * pad_len)
        Y.append(tags + [-1] * pad_len)
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(attn_masks, device=device),
        torch.tensor(Y, device=device),
    )

def pad_batch_old(batch): # this is the old one, swap back, if no improvement is seen with the new one 
    """Pads a batch of sentences to the same length and converts them to tensors.
    Words are padded with the index for <PAD> and tags are padded with -1 (which will be ignored in the loss function).
    Args:
        batch (list of tuples): A batch of sentences, where each sentence is a tuple of (word_indices, tag_indices)
    Returns:
        X (torch.Tensor): A tensor of shape (batch_size, max_seq_len) containing the padded word indices
        Y (torch.Tensor): A tensor of shape (batch_size, max_seq_len) containing the padded tag indices
    """
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
    """A simple BiLSTM-based POS tagger."""
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out)
        return logits
    

class BertBiLSTMTagger(nn.Module): # TODO
    def __init__(self, tagset_size, hidden_dim=128, freeze_bert=True): # hidden_dim og: 128
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        if freeze_bert:
            print("Freezing BERT parameters...")
            for p in self.bert.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out, _ = self.lstm(bert_out.last_hidden_state)
        return self.fc(out)

########################################
# Training Setup
########################################

print("\nSetting up model, optimizer, and loss function...")

model = BertBiLSTMTagger(len(tag2idx), hidden_dim=64, freeze_bert=True).to(device)
optimizer = optim.AdamW([
    {"params": model.bert.parameters(), "lr": 2e-5},
    {"params": model.lstm.parameters(), "lr": 1e-3},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
criterion = nn.CrossEntropyLoss(ignore_index=-1)

from transformers import get_linear_schedule_with_warmup

batch_size = 32
epochs = 10
early_stopping_patience = 2

total_steps = (len(train_data) // batch_size) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)


########################################
# Training Loop
########################################


def train(model, train_data, val_data, epochs=5, batch_size=32, patience=2):
    """Trains the BiLSTM POS tagger and performs early stopping on validation loss.
    
    Args:
        model (nn.Module): The BiLSTMTagger model to be trained
        train_data (list of list of tuples): The training data, where each sentence is a list of (word, tag) tuples
        val_data (list of list of tuples): The validation data, where each sentence is a list of (word, tag) tuples
        epochs (int): The number of epochs to train for
        batch_size (int): The number of sentences to include in each training batch
        patience (int): Number of epochs with no validation improvement before stopping
    """
    global_step = 0
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        total_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            encoded = [encode_sentence(s) for s in batch]
            input_ids, attn_mask, Y = pad_batch(encoded)          # was: X, Y = pad_batch(encoded)

            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)                  # was: logits = model(X)
            loss = criterion(logits.view(-1, logits.shape[-1]), Y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            global_step += 1

        avg_loss = total_loss / max(1, len(train_data) // batch_size)
        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f}")

        val_loss, val_acc = validate(model, val_data, global_step)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"New best validation loss: {best_val_loss:.4f} (acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for {epochs_without_improvement}/{patience} epochs "
                f"(best loss: {best_val_loss:.4f})"
            )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs with patience={patience}."
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model checkpoint from validation loss {best_val_loss:.4f}")


def validate(model, data, step, batch_size=32):
    """Evaluates the model on the validation data and logs the loss and accuracy to TensorBoard.
    
    Args:
        model (nn.Module): The BiLSTMTagger model to be evaluated
        data (list of list of tuples): The validation data, where each sentence is a list of (word, tag) tuples
        step (int): The current global step in training, used for logging to TensorBoard
        batch_size (int): The number of sentences to include in each evaluation batch
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            encoded = [encode_sentence(s) for s in batch]
            input_ids, attn_mask, Y = pad_batch(encoded)          # was: X, Y = pad_batch(encoded)

            logits = model(input_ids, attn_mask)                  # was: logits = model(X)
            loss = criterion(logits.view(-1, logits.shape[-1]), Y.view(-1))
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
    return avg_loss, acc


########################################
# Evaluation
########################################

def evaluate(model, data):
    """Evaluates the model on the test data and prints accuracy, F1 score, precision, and recall.
    
    Args:
        model (nn.Module): The BiLSTMTagger model to be evaluated
        data (list of list of tuples): The test data, where each sentence is a list of (word, tag) tuples
    """

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for sent in data:
            input_ids, attn_mask, gold_tags = encode_sentence(sent)  # now returns 3 things
            input_ids_t = torch.tensor([input_ids]).to(device)
            attn_mask_t = torch.tensor([attn_mask]).to(device)

            logits = model(input_ids_t, attn_mask_t)                 # was: model(X)
            preds = torch.argmax(logits, dim=-1)[0]

            for p, g in zip(preds, gold_tags):
                if g == -1:                     # skip padding/subword positions
                    continue
                total += 1
                correct += p.item() == g
                all_preds.append(p.item())
                all_golds.append(g)

    if total == 0:
        print("No gold tags to evaluate.")
        return -1
    acc = correct / total
    f1 = f1_score(all_golds, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_golds, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_golds, all_preds, average="weighted", zero_division=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

    # ---- Per-tag breakdown ----
    present_labels = sorted(set(all_golds) | set(all_preds))
    print("\nPer-tag classification report:")
    print(classification_report(
        all_golds, all_preds,
        labels=present_labels,
        target_names=[idx2tag[i] for i in present_labels],
        zero_division=0,
    ))


########################################
# Main Execution
########################################
writer = SummaryWriter(log_dir="runs/bnc_bilstm_pos_exploratory")

print("Training model...")
train(model, train_data, val_data, epochs=epochs, patience=early_stopping_patience)

print("\nEvaluating model on BNC...")
evaluate(model, test_data)
writer.close()

########################################
# Test with own testdata
########################################

# F1-Score
# Accuracy

# all testdata
print("\nEvaluating model on test sentences...")
evaluate(model, test_sentences)

# only essays
print("\nEvaluating model on essay test sentences...")
evaluate(model, essay_test)

# only picture description
print("\nEvaluating model on picture description test sentences...")
evaluate(model, picture_test)

# only chat
print("\nEvaluating model on chat test sentences...")
evaluate(model, chat_test)

# only short essay
print("\nEvaluating model on short essay test sentences...")
evaluate(model, short_essay_test)

# by age groups
#print("\nEvaluating model on Klasse 5 test sentences...")
#evaluate(model, test_sentences_klasse_5)

#print("\nEvaluating model on Klasse 6 test sentences...")
#evaluate(model, test_sentences_klasse_6)

#print("\nEvaluating model on Klasse 7 test sentences...")
#evaluate(model, test_sentences_klasse_7)

#print("\nEvaluating model on Klasse 8 test sentences...")
#evaluate(model, test_sentences_klasse_8)

#print("\nEvaluating model on Klasse 9 test sentences...")
#evaluate(model, test_sentences_klasse_9)

#print("\nEvaluating model on Klasse 10 test sentences...")
#evaluate(model, test_sentences_klasse_10)

#print("\nEvaluating model on Klasse 11 test sentences...")
#evaluate(model, test_sentences_klasse_11)

#print("\nEvaluating model on Klasse 12 test sentences...")
#evaluate(model, test_sentences_klasse_12)

writer.close()

########################################
# Create confusion matrix
########################################

def create_confusion_matrix(model, data, name):
    """Creates and saves a confusion matrix for the given model and test data using relative values and not the absolute values.
    
    Args:
        model (nn.Module): The BiLSTMTagger model to be evaluated
        data (list of list of tuples): The test data, where each sentence is a list of (word, tag) tuples
        name (str): The name to use for the saved confusion matrix image file
    """
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for sent in data:
            input_ids, attn_mask, gold_tags = encode_sentence(sent)  # now returns 3 things
            input_ids_t = torch.tensor([input_ids]).to(device)
            attn_mask_t = torch.tensor([attn_mask]).to(device)
            logits = model(input_ids_t, attn_mask_t)
            preds = torch.argmax(logits, dim=-1)[0]

            all_preds.extend(preds.cpu().numpy())
            all_golds.extend(gold_tags)

    # Get unique labels that actually appear in the data, sorted alphabetically by tag name
    unique_tags = sorted(set(all_golds) | set(all_preds), key=lambda x: idx2tag[x])
    # Map to tag names (already sorted alphabetically)
    label_names = [idx2tag[i] for i in unique_tags]
    # not absolute but relative numbers
    cm = confusion_matrix(all_golds, all_preds, labels=unique_tags, normalize="true")
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=False,
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
        cbar_kws={"shrink": 0.8},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Gold Labels", fontsize=12)
    plt.xticks(rotation=90, ha="center", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(name + " Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix_bilstm.png", dpi=150)
    plt.close()

#print("\nCreating confusion matrix for essay test sentences...")
#create_confusion_matrix(model, essay_test, "Essay Test Sentences")
#print("\nCreating confusion matrix for picture description test sentences...")
#create_confusion_matrix(model, picture_test, "Picture Description Test Sentences")
#print("\nCreating confusion matrix for chat test sentences...")
#create_confusion_matrix(model, chat_test, "Chat Test Sentences")
#print("\nCreating confusion matrix for short essay test sentences...")
#create_confusion_matrix(model, short_essay_test, "Short Essay Test Sentences")


def print_confused_sentences(model, data, target_tags, idx2tag, num_sentences=5):
    """Prints sentences from the test data where the model confused specific target tags with other tags.
    
    Args:
        model (nn.Module): The BiLSTMTagger model to be evaluated
        data (list of list of tuples): The test data, where each sentence is a list of (word, tag) tuples
        target_tags (set of str): The set of target tag names to look for confusion
        idx2tag (dict): A mapping from tag indices to tag names
        num_sentences (int): The number of confused sentences to print for each target tag
    """
    model.eval()
    confused_sentences = {tag: [] for tag in target_tags}

    with torch.no_grad():
        for sent in data:
            input_ids, attn_mask, gold_tags = encode_sentence(sent)  # now returns 3 things
            input_ids_t = torch.tensor([input_ids]).to(device)
            attn_mask_t = torch.tensor([attn_mask]).to(device)
            logits = model(input_ids_t, attn_mask_t)
            preds = torch.argmax(logits, dim=-1)[0]

            for word, gold_idx, pred_idx in zip(
                [w for w, _ in sent], gold_tags, preds.cpu().numpy()
            ):
                gold_tag = idx2tag.get(gold_idx, "<UNK>")
                pred_tag = idx2tag.get(pred_idx, "<UNK>")
                if gold_tag in target_tags and pred_tag != gold_tag:
                    if len(confused_sentences[gold_tag]) < num_sentences:
                        confused_sentences[gold_tag].append(
                            (word, gold_tag, pred_tag)
                        )

    for tag in target_tags:
        print(f"\nSentences where '{tag}' was confused:")
        for word, gold, pred  in confused_sentences[tag]:
            print(f"  Word: '{word}' | Gold: '{gold}' | Predicted: '{pred}'")

# print 5 sentences where tag "VVD", "AVP", "AJC", "VVG" or "VVN" was confused with another tag in essay test sentences
target_tags = {"VVD", "AVP", "AJC", "VVG", "VVN"}
print_confused_sentences(model, essay_test, target_tags, idx2tag, num_sentences=5)

# print 5 sentences where tag "AJC", "AV0", "AVP", "NP0", "PNI", "POS", VHZ", "VVD", "VVN" or "VVZ" was confused with another tag in chat sentences
target_tags_chat = {"AJC", "AV0", "AVP", "NP0", "PNI", "POS", "VHZ", "VVD", "VVN", "VVZ"}
print_confused_sentences(model, chat_test, target_tags_chat, idx2tag, num_sentences=5)

# print 5 sentences where tag "CJS", "CJT", "TO0", "VDI", "VHI", "VNN-AJ0", "VVN-VVD" was confused with another tag in picture description sentences
target_tags_picture = {"CJS", "CJT", "TO0", "VDI", "VHI", "VNN-AJ0", "VVN-VVD"}
print_confused_sentences(model, picture_test, target_tags_picture, idx2tag, num_sentences=5)

# print 5 sentences where tag "AJC", "CJT", "VHG", "VVB", "VVD", "VVG", "VVN" or "VVZ" was confused with another tag in short essay sentences
target_tags_short_essay = {"AJC", "CJT", "VHG", "VVB", "VVD", "VVG", "VVN", "VVZ"}
print_confused_sentences(model, short_essay_test, target_tags_short_essay, idx2tag, num_sentences=5)

# combine all "weak" tags from the different categories and print confused sentences for all of them
all_target_tags = {"CJT", "AJS", "DTQ", "DT0", "EX0", "ITJ", "NP0", "PNX", "POS", "PRP", "VBB", "VDI", "VHI", "VVG", "VVI", "VVN", "VVZ", "XX0", "ZZ0"}
print_confused_sentences(model, test_sentences, all_target_tags, idx2tag, num_sentences=5)