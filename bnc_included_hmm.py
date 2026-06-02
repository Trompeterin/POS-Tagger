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
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time

########################################
# Load BNC Data
########################################

xml_files = Path('BNC/Texts').rglob('*.xml')
all_sentences = []  # List of all sentences
all_tags = set()
all_words = set()

print("Load BNC data and extract sentences...")

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
    age_pattern = re.compile(r'(\s*Klasse\d+)')
    match = age_pattern.search(sentence)
    if match:
        age = match.group(1).strip()
        # extract the number from the age string
        age_number = int(re.search(r'\d+', age).group())
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

with open(test_path, 'r', encoding='utf-8') as f:
    empty = True # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            # ignore everything that the instructor says -> I_ZZ0
            if line.startswith("I_ZZ0"):
                continue
            if line.startswith("S_ZZ0"):
                tokens = tokens[2:]  # remove the first two tokens which is the metadata about who is speaking
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                test_sentences_chat.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True

with open(test_path_2, 'r', encoding='utf-8') as f:
    empty = True # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                test_sentences_essay.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True

# Data that was used to test CLAWS etc.
with open(test_path_3, 'r', encoding='utf-8') as f:
    empty = True # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
                    sentence.append((word, tag))
            if sentence:
                test_sentences.append(sentence)
                store_sentence_by_age_group(sentence, age_group)
        else:
            empty = True

with open(test_path_4, 'r', encoding='utf-8') as f:
    empty = True # so that the first line with metadata is correctly identified as such
    age_group = None
    for line in f:
        if line.strip():
            # after each empty line the line starts with meta data about age group and school type of the following sentences
            if empty:
                age_group = extract_age_group(line)
                empty = False
                if age_group is not None:
                    continue # this is not part of the sentences and should not be added
            tokens = line.strip().split()
            sentence = []
            for token in tokens:
                if '_' in token and len(token.rsplit('_', 1)) == 2:
                    word, tag = token.rsplit('_', 1)
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

# split own data into train and test sentences randomly by category
random.shuffle(test_sentences_essay)
random.shuffle(test_sentences_chat)
random.shuffle(test_sentences_picture)
random.shuffle(test_sentences_short_essay)

# Calculate 80/20 split indices for each category
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

# print total train data sentences from own data
print(f"Total train sentences from own data: {len(essay_train) + len(chat_train) + len(picture_train) + len(short_essay_train):,}")

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
# Setup
########################################
tagged_sents = sentences_A
random.shuffle(tagged_sents)

# Shuffle and split
split = int(len(tagged_sents) * 0.8)
train_data = tagged_sents[:split]
test_data  = tagged_sents[split:]

train_data.extend(essay_train)
train_data.extend(chat_train)
train_data.extend(picture_train)
train_data.extend(short_essay_train)

# shuffel train data after adding the test sentences to it
random.shuffle(train_data)

print("Setup completed. Starting training...")

# -----------------------------
# TRAIN: count frequencies
# -----------------------------

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

# -----------------------------
# VITERBI DECODER
# -----------------------------
def viterbi(words):
    """Viterbi algorithm to find the most likely sequence of tags for a given sequence of words.
    Args:
        words (list of str): The input sequence of words.
    Returns:
        list of str: The most likely sequence of tags corresponding to the input words."""
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

# Evaluation
print("Start evaluation on BNC test set...")
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
precision = precision_score(all_gold_tags, all_pred_tags, average='weighted')
recall = recall_score(all_gold_tags, all_pred_tags, average='weighted')
print(f"Accuracy on BNC test set: {accuracy:.4f}")
print(f"F1-Score on BNC test set: {f1:.4f}")
print(f"Precision on BNC test set: {precision:.4f}")
print(f"Recall on BNC test set: {recall:.4f}")

########################################
# Test with own testdata
########################################

# load your own test sentences in the same format as train_data (list of (word, tag) pairs)

# F1-Score
# Accuracy

def evaluate_on_test_sentences(model, test_sentences):
    """Evaluates the given model on the provided test sentences and prints accuracy, F1-Score, Precision, and Recall.
    Args:
        model (function): The POS tagging model to be evaluated, which takes a list of words and returns a list of predicted tags.
        test_sentences (list of list of tuples): The test sentences, where each sentence is a list of (word, tag) tuples.
    """
    all_golds = []
    all_preds = []
    for sent in test_sentences:
        words = [w for (w, _) in sent]
        gold_tags = [t for (_, t) in sent]
        pred_tags = model(words)
        all_golds.extend(gold_tags)
        all_preds.extend(pred_tags)
    if len(all_golds) == 0:
        print("No gold tags found in test sentences. Cannot compute accuracy or F1-Score.")
        return
    acc = sum(g == p for g, p in zip(all_golds, all_preds)) / len(all_golds)
    f1 = f1_score(all_golds, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_golds, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_golds, all_preds, average='weighted', zero_division=0)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1 score: {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")

# F1-Score
# Accuracy

# all testdata 
print("\nEvaluating model on test sentences...")
evaluate_on_test_sentences(viterbi, test_sentences)

# only essays 
print("\nEvaluating model on essay test sentences...")
evaluate_on_test_sentences(viterbi, essay_test)

# only picture description
print("\nEvaluating model on picture description test sentences...")
evaluate_on_test_sentences(viterbi, picture_test)

# only chat
print("\nEvaluating model on chat test sentences...")
evaluate_on_test_sentences(viterbi, chat_test)

# only short essay
print("\nEvaluating model on short essay test sentences...")
evaluate_on_test_sentences(viterbi, short_essay_test)

# by age groups 
#print("\nEvaluating model on Klasse 5 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_5)

#print("\nEvaluating model on Klasse 6 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_6)

#print("\nEvaluating model on Klasse 7 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_7)

#print("\nEvaluating model on Klasse 8 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_8)

#print("\nEvaluating model on Klasse 9 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_9)

#print("\nEvaluating model on Klasse 10 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_10)

#print("\nEvaluating model on Klasse 11 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_11)

#print("\nEvaluating model on Klasse 12 test sentences...")
#evaluate_on_test_sentences(viterbi, test_sentences_klasse_12)

########################################
# Create confusion matrix
########################################

# Define unique labels and label names from tags
unique_labels = sorted(tags)
label_names = unique_labels

def create_confusion_matrix(model, data, name): 
    """Creates and saves a confusion matrix for the given model and test data.
    Args:        
        model (function): The POS tagging model to be evaluated, which takes a list of words and returns a list of predicted tags.
        data (list of list of tuples): The test sentences, where each sentence is a list of (word, tag) tuples.
        name (str): The name to be used for the title of the confusion matrix and the filename when saving the plot.
    """
    all_preds = []
    all_golds = []
    # get predictions and gold tags for all sentences in data
    for sent in data:
        words = [w for (w, _) in sent]
        gold_tags = [t for (_, t) in sent]
        pred_tags = model(words)
        all_golds.extend(gold_tags)
        all_preds.extend(pred_tags)
    # we want relative values and not absolute values, so we normalize the confusion matrix by the true labels (gold tags)
    cm = confusion_matrix(all_golds, all_preds, labels=unique_labels, normalize='true')
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, xticklabels=label_names, yticklabels=label_names, cmap='Blues', cbar_kws={'shrink': 0.8})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Gold Labels', fontsize=12)
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(name + " Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix_hmm.png", dpi=150)
    plt.close()