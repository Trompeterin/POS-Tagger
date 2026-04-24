import nltk
from collections import defaultdict, Counter
import math
import random
import random
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import treebank
import os
from pathlib import Path
import lxml.etree as ET
import tensorflow as tf
import datetime
from sklearn.metrics import f1_score
import time

########################################
# Load BNC Data
########################################
start_time = time.time()

xml_files = Path('BNC/Texts').rglob('*.xml')
all_sentences = []  # List of all sentences
all_tags = set()
all_words = set()

print("Loading BNC data from XML files...")

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
pct = 10  # percentage of sentences to use           
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


print(f"Loaded {len(test_sentences):,} sentences from test data.")

########################################
# Setup
########################################
tagged_sents = sentences_A
random.shuffle(tagged_sents)

# Shuffle and split
split = int(len(tagged_sents) * 0.8)
train_data = tagged_sents[:split]
test_data  = tagged_sents[split:]
# print time it took to load data and split into train and test
end_time = time.time()
print(f"Time taken to load BNC data and split into train/test: {end_time - start_time:.2f} seconds")

print("Setup completed. Starting training...")

# -----------------------------
# TRAIN: count frequencies
# -----------------------------

start_time = time.time()

print("Start counting frequencies...")

tags = set()
vocab = set()

start_counts = Counter()
trans_counts = defaultdict(Counter)
emit_counts  = defaultdict(Counter)
tag_counts   = Counter()

for sent in train_data:
    prev_tag = None
    for i, (word, tag) in enumerate(sent):
        tags.add(tag)
        vocab.add(word)
        tag_counts[tag] += 1
        emit_counts[tag][word] += 1

        if i == 0:
            start_counts[tag] += 1
        if prev_tag is not None:
            trans_counts[prev_tag][tag] += 1
        prev_tag = tag

tags = list(tags)
V = len(vocab)
T = len(tags)

# Add-k smoothing
k = 1.0

print("Start precomputing log-probabilities...")

# Precompute log-probabilities
start_logp = {}
trans_logp = {}
emit_logp  = {}

total_starts = sum(start_counts.values()) + k * T
for tag in tags:
    start_logp[tag] = math.log((start_counts[tag] + k) / total_starts)

for prev_tag in tags:
    trans_logp[prev_tag] = {}
    total = sum(trans_counts[prev_tag].values()) + k * T
    for tag in tags:
        trans_logp[prev_tag][tag] = math.log((trans_counts[prev_tag][tag] + k) / total)

for tag in tags:
    emit_logp[tag] = {}
    total = sum(emit_counts[tag].values()) + k * (V + 1)
    for word in vocab:
        emit_logp[tag][word] = math.log((emit_counts[tag][word] + k) / total)
    # unknown token
    emit_logp[tag]["<UNK>"] = math.log(k / total)

end_time = time.time()
print(f"Time taken to count frequencies and precompute log-probabilities: {end_time - start_time:.2f} seconds")

# -----------------------------
# VITERBI DECODER
# -----------------------------
def viterbi(words):
    N = len(words)
    dp = [{} for _ in range(N)]
    backpointer = [{} for _ in range(N)]

    # Initialization
    for tag in tags:
        emit_p = emit_logp[tag].get(words[0], emit_logp[tag]["<UNK>"])
        dp[0][tag] = start_logp[tag] + emit_p
        backpointer[0][tag] = None

    # Recursion
    for t in range(1, N):
        for curr_tag in tags:
            best_prob, best_prev = float('-inf'), None
            emit_p = emit_logp[curr_tag].get(words[t], emit_logp[curr_tag]["<UNK>"])
            for prev_tag in tags:
                prob = dp[t-1][prev_tag] + trans_logp[prev_tag][curr_tag] + emit_p
                if prob > best_prob:
                    best_prob = prob
                    best_prev = prev_tag
            dp[t][curr_tag] = best_prob
            backpointer[t][curr_tag] = best_prev

    # Termination
    # choose best final tag
    best_final = max(dp[N-1], key=dp[N-1].get)
    tags_out = [best_final]

    for t in range(N-1, 0, -1):
        tags_out.append(backpointer[t][tags_out[-1]])
    tags_out.reverse()
    return tags_out

start_time = time.time()
# Evaluation
print("Start evaluation on test set...")
total_words = 0
correct_tags = 0
all_gold_tags = []
all_pred_tags = []

for sent in test_data:
    words = [w for (w, _) in sent]
    gold_tags = [t for (_, t) in sent]
    pred_tags = viterbi(words)
    all_gold_tags.extend(gold_tags)
    all_pred_tags.extend(pred_tags)
    for gt, pt in zip(gold_tags, pred_tags):
        total_words += 1
        if gt == pt:
            correct_tags += 1

accuracy = correct_tags / total_words
f1 = f1_score(all_gold_tags, all_pred_tags, average='weighted')
print(f"Accuracy on test set: {accuracy:.4f}")
print(f"F1-Score on test set: {f1:.4f}")

end_time = time.time()
print(f"Time taken for evaluation: {end_time - start_time:.2f} seconds")

########################################
# Test with own testdata
########################################

# load your own test sentences in the same format as train_data (list of (word, tag) pairs)

# F1-Score
# Accuracy

def evaluate_on_test_sentences(model, test_sentences):
    all_golds = []
    all_preds = []
    for sent in test_sentences:
        words = [w for (w, _) in sent]
        gold_tags = [t for (_, t) in sent]
        pred_tags = model(words)
        all_golds.extend(gold_tags)
        all_preds.extend(pred_tags)

    acc = sum(g == p for g, p in zip(all_golds, all_preds)) / len(all_golds)
    f1 = f1_score(all_golds, all_preds, average='weighted', zero_division=0)
    print(f"Test accuracy on own test sentences: {acc:.4f}")
    print(f"Test F1 score on own test sentences: {f1:.4f}")

evaluate_on_test_sentences(viterbi, test_sentences)

print("All done!")