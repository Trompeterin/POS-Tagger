import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
import lxml.etree as ET
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score, confusion_matrix
)
import re
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup


########################################
# Configuration
########################################
@dataclass
class Config:
    """Configuration for BiLSTM POS Tagger training and evaluation."""
    # Paths
    bnc_path: str = "Code/BNC/Texts"
    test_data_dir: str = "POS-Tagger/POS-Tagging-Testdaten"
    model_save_path: str = "bilstm_pos_tagger_bnc.pth"
    tensorboard_log_dir: str = "runs/bnc_bilstm_pos_exploratory"
    false_predictions_path: str = "false_predictions.txt"
    
    # Data loading
    bnc_sample_pct: int = 5  # percentage of BNC sentences to use
    train_val_split: float = 0.9  # 90% train, 10% val from BNC
    val_test_split: float = 0.95  # 95% train+val, 5% test from BNC
    own_data_train_split: float = 0.8  # 80% train, 20% test from own data
    own_data_val_split: float = 0.1  # 10% val from own train data
    
    # Model
    model_name: str = "bert-base-multilingual-cased"
    max_seq_length: int = 128
    hidden_dim: int = 64
    freeze_bert: bool = True
    
    # Training
    batch_size: int = 32
    epochs: int = 5
    early_stopping_patience: int = 1
    learning_rate_bert: float = 2e-5
    learning_rate_lstm: float = 1e-3
    learning_rate_fc: float = 1e-3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    validation_step_interval: int = 1000
    
    # Evaluation
    random_seed: int = 42

########################################
# Checkpoint handling
########################################
def save_checkpoint_safely(model: nn.Module, idx2tag: dict[int, str], tag2idx: dict[str, int], path: str | Path):
    """Persist a model checkpoint and verify it immediately by reloading it."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")

    payload = {
        "model_state_dict": model.state_dict(),
        "idx2tag": idx2tag,
        "tag2idx": tag2idx,
    }

    torch.save(payload, tmp_path)

    try:
        loaded = torch.load(str(tmp_path), map_location="cpu")
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected a dict checkpoint, got {type(loaded).__name__}.")
        required = {"model_state_dict", "idx2tag", "tag2idx"}
        missing = sorted(required - set(loaded.keys()))
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")
        # ensure model state can be reloaded into a blank model instance
        model.load_state_dict(loaded["model_state_dict"])
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Checkpoint verification failed for '{save_path}'. "
            "The file is corrupt or incomplete; retrain and save again."
        ) from exc

    tmp_path.replace(save_path)
    print(f"Model saved and verified to {save_path}")

########################################
# Load BNC Data
########################################
def load_bnc_data(config: Config):
    """Loads a random sample of sentences from the BNC corpus, along with their POS tags.
    
    Args:
        config (Config): Configuration object containing paths and sampling parameters
        
    Returns:
        tuple: (sentences, all_words, all_tags) from the BNC corpus
    """
    print("Loading BNC data...")
    
    xml_files = list(Path(config.bnc_path).rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {config.bnc_path}")
    
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # First pass: count non-empty sentences
    total_sentence_count = 0
    for xml_file in xml_files:
        try:
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
        except ET.ParseError as e:
            print(f"Warning: Failed to parse {xml_file}: {e}")
            continue

    sample_size = max(1, total_sentence_count * config.bnc_sample_pct // 100)

    # Second pass: reservoir sample sentences
    sampled_sentences = []
    seen_sentences = 0
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for sentence in root.findall(".//s"):
                word_tag_pairs = []
                for word in sentence.findall(".//w"):
                    word_text = word.text
                    pos_tag = word.get("c5")
                    if word_text and pos_tag:
                        # if tag of form XXX-YYY keep only XXX
                        if pos_tag and "-" in pos_tag:
                            pos_tag = pos_tag.split("-")[0]
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
        except ET.ParseError as e:
            print(f"Warning: Failed to parse {xml_file}: {e}")
            continue

    all_words = {word for sent in sampled_sentences for word, _ in sent}
    all_tags = {tag for sent in sampled_sentences for _, tag in sent}

    print(
        f"Using {len(sampled_sentences):,} sentences out of {total_sentence_count:,} total "
        f"sentences in BNC ({len(sampled_sentences)/max(1, total_sentence_count):.2%})"
    )
    print(f"Total unique words: {len(all_words):,}")
    print(f"Total unique tags: {len(all_tags)}")

    return sampled_sentences, all_words, all_tags

########################################
# Load Tagged Testdata and Split into Age Groups
########################################

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


def store_sentence_by_age_group(sentence, age_group, age_group_sentences: dict):
    """
    Stores the given sentence in the appropriate list based on the age group.
    Args:
        sentence (list of tuples): The sentence to be stored, where each tuple contains a word and its corresponding tag.
        age_group (int): The age group number extracted from the metadata.
        age_group_sentences (dict): Dictionary with age group keys (5-12) containing lists of sentences.
    """
    if age_group in age_group_sentences:
        age_group_sentences[age_group].append(sentence)

########################################
# Load Tagged Testdata
########################################
def load_test_sentences_from_file(test_path, test_sentences, age_group_sentences=None):
    """Load test sentences from a file, optionally sorting by age group.
    
    Args:
        test_path (str): Path to the test file
        test_sentences (list): List to append all sentences to
        age_group_sentences (dict|None): Dictionary with age group keys (5-12) for grouping by age
        
    Returns:
        list: Sentences loaded from this specific file
        
    Raises:
        FileNotFoundError: If the test file doesn't exist
    """
    if not Path(test_path).exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    print(f"Loading test sentences from {Path(test_path).name}...")
    test_sentences_subgroup = []
    age_group = None
    empty = True
    
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                # First non-empty line after empty line contains metadata
                if empty:
                    age_group = extract_age_group(line)
                    empty = False
                    if age_group is not None:
                        continue
                
                # Skip instructor lines
                if line.startswith("I_ZZ0"):
                    continue
                
                tokens = line.strip().split()
                if line.startswith("S_ZZ0"):
                    tokens = tokens[2:]  # Remove speaker metadata
                
                sentence = []
                for token in tokens:
                    if "_" in token and len(token.rsplit("_", 1)) == 2:
                        word, tag = token.rsplit("_", 1)
                        sentence.append((word, tag))
                
                if sentence:
                    test_sentences.append(sentence)
                    test_sentences_subgroup.append(sentence)
                    if age_group_sentences is not None:
                        store_sentence_by_age_group(sentence, age_group, age_group_sentences)
            else:
                empty = True
    
    return test_sentences_subgroup


def encode_sentence(sent, tokenizer, tag2idx, max_seq_length):
    """Encode a sentence into token ids, attention mask and label ids.

    Args:
        sent (list): List of (word, tag) tuples
        tokenizer: HuggingFace tokenizer
        tag2idx (dict): Mapping from tag strings to indices
        max_seq_length (int): Maximum sequence length for truncation
        
    Returns:
        tuple: (input_ids, attention_mask, label_ids)
    """
    if tokenizer is None:
        raise RuntimeError("Tokenizer must be provided.")
    if tag2idx is None:
        tag2idx = {"<UNK>": 0}

    words = [w for w, _ in sent]
    tags = [t for _, t in sent]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
    )
    word_ids = encoding.word_ids()

    label_ids = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-1)
        elif word_idx != prev_word_idx:
            label_ids.append(tag2idx.get(tags[word_idx], tag2idx.get("<UNK>", 0)))
        else:
            label_ids.append(-1)
        prev_word_idx = word_idx

    return encoding["input_ids"], encoding["attention_mask"], label_ids

def pad_batch(batch, tokenizer, device, max_seq_length=None):
    """Pads a batch of sentences to the same length and converts them to tensors.
    
    Args:
        batch (list): List of tuples (input_ids, attention_mask, labels)
        tokenizer: HuggingFace tokenizer with pad_token_id attribute
        device (torch.device): Device to place tensors on
        max_seq_length (int|None): For debug logging (optional)
        
    Returns:
        tuple: (input_ids_tensor, attention_masks_tensor, labels_tensor)
    """
    lengths = [len(ids) for ids, _, _ in batch]
    max_len = max(lengths)
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


########################################
# Model Definition
########################################

class BertBiLSTMTagger(nn.Module):
    """BiLSTM-based POS tagger with BERT embeddings."""
    def __init__(self, tagset_size, model_name, hidden_dim=128, freeze_bert=True): 
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        print("Freezing BERT parameters...")
        for p in self.bert.parameters():
            p.requires_grad = False

        if not freeze_bert:
            print("Unfreezing BERT parameters of last 2 layers...")
            for p in list(self.bert.parameters())[-4:]:  # Unfreeze last 2 layers
                p.requires_grad = True

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
# Training Loop
########################################


def train(model, train_data, val_data, optimizer, scheduler, criterion, tokenizer, 
          tag2idx, device, config: Config, writer):
    """Trains the BiLSTM POS tagger and performs early stopping on validation loss.
    
    Args:
        model (nn.Module): The BertBiLSTMTagger model to be trained
        train_data (list): Training data sentences
        val_data (list): Validation data sentences
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        criterion: Loss function
        tokenizer: HuggingFace tokenizer
        tag2idx (dict): Tag to index mapping
        device (torch.device): Device to use for training
        config (Config): Configuration object
        writer: TensorBoard writer
    """
    global_step = 0
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(config.epochs):
        model.train()
        random.shuffle(train_data)
        total_loss = 0

        for i in range(0, len(train_data), config.batch_size):
            batch = train_data[i : i + config.batch_size]
            encoded = [encode_sentence(s, tokenizer, tag2idx, config.max_seq_length) 
                      for s in batch]
            input_ids, attn_mask, Y = pad_batch(encoded, tokenizer, device)

            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss = criterion(logits.view(-1, logits.shape[-1]), Y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if global_step % config.validation_step_interval == 0 and global_step > 0:
                print(f"Epoch {epoch+1} | Step {global_step} | Training Loss: {loss.item():.4f}")
                val_loss, val_acc = validate(model, val_data, optimizer, criterion, 
                                            tokenizer, tag2idx, device, config, 
                                            global_step, writer)
                model.train()  # switch back to training mode after validation
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    best_epoch = epoch
                    print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
                elif best_epoch != epoch:
                    epochs_without_improvement += 1
                    print(
                        f"No validation improvement for {epochs_without_improvement} epochs "
                        f"(best loss: {best_val_loss:.4f})"
                    )

            total_loss += loss.item()
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            global_step += 1

        avg_loss = total_loss / max(1, len(train_data) // config.batch_size)
        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f}")

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs with patience={config.early_stopping_patience}."
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model checkpoint from validation loss {best_val_loss:.4f}")


def validate(model, data, optimizer, criterion, tokenizer, tag2idx, device, 
             config: Config, step, writer):
    """Evaluates the model on the validation data and logs metrics to TensorBoard.
    
    Args:
        model (nn.Module): The BertBiLSTMTagger model to be evaluated
        data (list): Validation data sentences
        optimizer: Optimizer (not used here, but passed for consistency)
        criterion: Loss function
        tokenizer: HuggingFace tokenizer
        tag2idx (dict): Tag to index mapping
        device (torch.device): Device to use
        config (Config): Configuration object
        step (int): Current global step in training
        writer: TensorBoard writer
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for i in range(0, len(data), config.batch_size):
            batch = data[i : i + config.batch_size]
            encoded = [encode_sentence(s, tokenizer, tag2idx, config.max_seq_length) 
                      for s in batch]
            input_ids, attn_mask, Y = pad_batch(encoded, tokenizer, device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits.view(-1, logits.shape[-1]), Y.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = Y != -1
            correct += (preds[mask] == Y[mask]).sum().item()
            total += mask.sum().item()

    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / max(1, len(data) // config.batch_size)
    print(f"Validation | loss: {avg_loss:.4f}, acc: {acc:.4f}")
    writer.add_scalar("Loss/validation", avg_loss, step)
    writer.add_scalar("Accuracy/validation", acc, step)
    return avg_loss, acc


########################################
# Evaluation
########################################

def evaluate(
    model,
    data,
    tokenizer,
    tag2idx,
    idx2tag,
    device,
    config: Config,
    evaluation_name: str = "EVALUATION",
    false_predictions_path: str | Path | None = None,
):
    """Evaluates the model on test data and prints accuracy, F1 score, precision, and recall.
    
    Args:
        model (nn.Module): The BertBiLSTMTagger model to be evaluated
        data (list): Test data sentences
        tokenizer: HuggingFace tokenizer
        tag2idx (dict): Tag to index mapping
        idx2tag (dict): Index to tag reverse mapping
        device (torch.device): Device to use
        config (Config): Configuration object
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_golds = []
    false_predictions = []

    with torch.no_grad():
        for sent in data:
            input_ids, attn_mask, gold_tags = encode_sentence(sent, tokenizer, tag2idx, 
                                                              config.max_seq_length)
            encoding = tokenizer(
                [word for word, _ in sent],
                is_split_into_words=True,
                truncation=True,
                max_length=config.max_seq_length,
            )
            input_ids_t = torch.tensor([input_ids]).to(device)
            attn_mask_t = torch.tensor([attn_mask]).to(device)

            logits = model(input_ids_t, attn_mask_t)
            preds = torch.argmax(logits, dim=-1)[0]

            sentence_tokens = []
            sentence_golds = []
            sentence_preds = []
            previous_word_id = None

            for position, (p, g) in enumerate(zip(preds, gold_tags)):
                if g == -1:  # skip padding/subword positions
                    continue
                total += 1
                correct += p.item() == g
                all_preds.append(p.item())
                all_golds.append(g)

                word_id = encoding.word_ids()[position]
                if word_id is not None and word_id != previous_word_id:
                    sentence_tokens.append(sent[word_id][0])
                    sentence_golds.append(idx2tag.get(g, "<UNK>"))
                    sentence_preds.append(idx2tag.get(p.item(), "<UNK>"))
                previous_word_id = word_id

            if sentence_golds and sentence_golds != sentence_preds:
                false_predictions.append(
                    (
                        " ".join(sentence_tokens),
                        sentence_golds,
                        sentence_preds,
                    )
                )

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

    if false_predictions_path is not None:
        output_path = Path(false_predictions_path)
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(f"{evaluation_name.upper()}\n\n")
            for sentence, gold_tags, predicted_tags in false_predictions:
                output_file.write(f"Sentence: {sentence}\n")
                output_file.write(f"Gold tags: {' '.join(gold_tags)}\n")
                output_file.write(f"Predicted tags: {' '.join(predicted_tags)}\n\n")
            output_file.write("-" * 108 + "\n\n")


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
# Inference Function and Helper Functions
########################################

def preprocess_sentence(sentence):
    """Preprocesses a sentence list of tokens into a list of tokens with correct splitting.

    Args:
        sentence: Input sentence list of tokens (words)

    Returns:
        list[str]: List of tokens
    """
    # then split negations like doesn't into does n't and it's into it 's
    processed_tokens = []
    for token in sentence:
        if re.match(r"\w+n't$", token):
            processed_tokens.append(token[:-3])
            processed_tokens.append("n't")
        elif re.match(r"\w+'\w+$", token):
            processed_tokens.append(token[:-2])
            processed_tokens.append(token[-2:])
        else:
            processed_tokens.append(token)
    return processed_tokens

def inference(model, sentence, tokenizer, tag2idx, idx2tag=None, device=None, max_seq_length=128):
    """Perform inference on a token sequence using explicit resources.

    Args:
        model (nn.Module): The loaded model
        sentence (list[str]): Token sequence (words)
        tokenizer: HuggingFace tokenizer
        tag2idx (dict): Mapping tag->index used during training
        idx2tag (dict|None): Reverse mapping index->tag
        device (torch.device|None): Device to run inference on
        max_seq_length (int): Maximum sequence length

    Returns:
        list[str]: Predicted tag strings for input tokens
        list[str]: Used tokens during inference
    """
    if idx2tag is None:
        idx2tag = {v: k for k, v in tag2idx.items()}

    model_device = device if device is not None else next(model.parameters()).device
    model.to(model_device)
    model.eval()
    
    with torch.no_grad():
        sent = [(w, "<UNK>") for w in preprocess_sentence(sentence)]
        input_ids, attn_mask, label_ids = encode_sentence(sent, tokenizer, tag2idx, max_seq_length)
        input_ids_t = torch.tensor([input_ids], device=model_device)
        attn_mask_t = torch.tensor([attn_mask], device=model_device)

        logits = model(input_ids_t, attn_mask_t)
        preds = torch.argmax(logits, dim=-1)[0]

        # tokens corresponding to input_ids
        tokens_all = tokenizer.convert_ids_to_tokens(input_ids)

        # merge WordPiece subtokens back to original tokens
        merged_tokens = []
        orig_to_merged = []
        for tok in tokens_all:
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                orig_to_merged.append(None)
                continue

            if tok.startswith("##") and merged_tokens:
                merged_tokens[-1] += tok[2:]
                orig_to_merged.append(len(merged_tokens) - 1)
                continue

            if tok == "'" and merged_tokens:
                merged_tokens[-1] += "'"
                orig_to_merged.append(len(merged_tokens) - 1)
                continue

            if tok in {"s", "t", "m", "d", "re", "ve", "ll"} and merged_tokens and merged_tokens[-1].endswith("'"):
                merged_tokens[-1] += tok
                orig_to_merged.append(len(merged_tokens) - 1)
                continue

            merged_tokens.append(tok)
            orig_to_merged.append(len(merged_tokens) - 1)

        predicted_tags = []
        tokens = []
        for position, pred in enumerate(preds):
            if label_ids[position] == -1:
                continue
            merged_idx = orig_to_merged[position]
            if merged_idx is None:
                continue
            token = merged_tokens[merged_idx]
            if not tokens or tokens[-1] != token:
                tokens.append(token)
                predicted_tags.append(idx2tag.get(pred.item(), "<UNK>"))

    return predicted_tags, tokens


########################################
# Setup Helper Functions
########################################

def setup_device():
    """Detects and sets the available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def load_test_data(config: Config):
    """Loads and processes all test data from files.
    
    Args:
        config (Config): Configuration object
        
    Returns:
        dict: Dictionary containing processed test data for different categories
    """
    test_data_paths = {
        "essay": Path(config.test_data_dir) / "Essay 2.txt",
        "chat": Path(config.test_data_dir) / "Digital Chat Daten Annotiert.txt",
        "picture": Path(config.test_data_dir) / "Picture Description.txt",
        "short_essay": Path(config.test_data_dir) / "Short Essay.txt",
        "claws": Path(config.test_data_dir) / "CLAWS Verbessert Neu.txt",
    }
    
    # Initialize age group dictionary
    age_group_sentences = {k: [] for k in range(5, 13)}
    
    # Load all test files
    test_sentences_by_category = {}
    all_test_sentences = []
    
    for category, path in test_data_paths.items():
        test_sentences_by_category[category] = load_test_sentences_from_file(
            str(path), all_test_sentences, age_group_sentences
        )
    
    return {
        "all": all_test_sentences,
        "by_category": test_sentences_by_category,
        "by_age_group": age_group_sentences,
    }


def split_own_data(test_data_dict, config: Config):
    """Splits own test data into train/val/test splits.
    
    Args:
        test_data_dict (dict): Dictionary from load_test_data()
        config (Config): Configuration object
        
    Returns:
        dict: Split data with train, val, and test for each category
    """
    result = {}
    
    for category, sentences in test_data_dict["by_category"].items():
        random.shuffle(sentences)
        split_idx_train_test = int(config.own_data_train_split * len(sentences))
        split_idx_train_val = int(config.own_data_val_split * len(sentences[:split_idx_train_test]))
        
        train_data = sentences[:split_idx_train_test]
        test_data = sentences[split_idx_train_test:]
        
        random.shuffle(train_data)
        val_data = train_data[:split_idx_train_val]
        train_data = train_data[split_idx_train_val:]
        
        result[category] = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }
    
    return result


def build_vocabulary(train_data):
    """Builds tag index mappings from training data.
    
    Args:
        train_data (list): Training sentences
        
    Returns:
        tuple: (tag2idx, idx2tag) dictionaries
    """
    tag2idx = {"<UNK>": 0}
    for sent in train_data:
        for word, tag in sent:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    idx2tag = {v: k for k, v in tag2idx.items()}
    return tag2idx, idx2tag


def setup_model_and_training(tag2idx, train_data, device, config: Config):
    """Creates model, optimizer, scheduler, and criterion.
    
    Args:
        tag2idx (dict): Tag to index mapping
        train_data (list): Training data for calculating total steps
        device (torch.device): Device for training
        config (Config): Configuration object
        
    Returns:
        dict: Dictionary with model, optimizer, scheduler, criterion, tokenizer
    """
    print("\nSetting up model, optimizer, and loss function...")
    
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    model = BertBiLSTMTagger(
        len(tag2idx),
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        freeze_bert=config.freeze_bert
    ).to(device)
    
    optimizer = optim.AdamW([
        {"params": model.bert.parameters(), "lr": config.learning_rate_bert},
        {"params": model.lstm.parameters(), "lr": config.learning_rate_lstm},
        {"params": model.fc.parameters(), "lr": config.learning_rate_fc},
    ])
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Calculate total steps for scheduler
    total_steps = (len(train_data) // config.batch_size) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
    }


def run_training_and_evaluation(config: Config):
    """Main training and evaluation pipeline.
    
    Args:
        config (Config): Configuration object
    """
    # Setup device
    device = setup_device()
    
    # Load BNC data
    print("\n" + "="*50)
    print("Loading BNC data...")
    print("="*50)
    bnc_sentences, _, _ = load_bnc_data(config)
    
    # Load and split own test data
    print("\n" + "="*50)
    print("Loading own test data...")
    print("="*50)
    test_data_dict = load_test_data(config)
    own_data_splits = split_own_data(test_data_dict, config)
    
    # Prepare training/val/test data
    split1 = int(config.train_val_split * len(bnc_sentences))
    split2 = int(config.val_test_split * len(bnc_sentences))
    
    train_data = bnc_sentences[:split1]
    val_data = bnc_sentences[split1:split2]
    test_data_bnc = bnc_sentences[split2:]
    
    # Add own data to training and validation
    for category_data in own_data_splits.values():
        train_data.extend(category_data["train"])
        val_data.extend(category_data["val"])
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # Print data statistics
    print(f"\n{'='*50}")
    print("Data Statistics:")
    print(f"{'='*50}")
    print(f"Train sentences: {len(train_data):,}")
    print(f"Validation sentences: {len(val_data):,}")
    print(f"Test sentences (BNC): {len(test_data_bnc):,}")
    
    for category, splits in own_data_splits.items():
        print(f"Test sentences ({category}): {len(splits['test']):,}")
    
    # Build vocabulary
    tag2idx, idx2tag = build_vocabulary(train_data)
    print(f"\nUnique tags: {len(tag2idx)}")
    
    # Setup model and training components
    training_components = setup_model_and_training(tag2idx, train_data, device, config)
    model = training_components["model"]
    tokenizer = training_components["tokenizer"]
    optimizer = training_components["optimizer"]
    scheduler = training_components["scheduler"]
    criterion = training_components["criterion"]
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
    
    # Train model
    print(f"\n{'='*50}")
    print("Training model...")
    print(f"{'='*50}")
    train(model, train_data, val_data, optimizer, scheduler, criterion, tokenizer,
          tag2idx, device, config, writer)
    
    # Save model and verify the saved checkpoint is readable before continuing.
    save_checkpoint_safely(model, idx2tag, tag2idx, config.model_save_path)

    false_predictions_path = Path(config.false_predictions_path)
    false_predictions_path.write_text("", encoding="utf-8")
    
    # Evaluate on all test sets
    print(f"\n{'='*50}")
    print("Model Evaluation:")
    print(f"{'='*50}")
    
    print("\nEvaluating on BNC test data...")
    evaluate(
        model,
        test_data_bnc,
        tokenizer,
        tag2idx,
        idx2tag,
        device,
        config,
        evaluation_name="BNC",
        false_predictions_path=false_predictions_path,
    )
    
    for category, splits in own_data_splits.items():
        print(f"\nEvaluating on {category.upper()} test data...")
        evaluate(
            model,
            splits["test"],
            tokenizer,
            tag2idx,
            idx2tag,
            device,
            config,
            evaluation_name=category,
            false_predictions_path=false_predictions_path,
        )
    
    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    config = Config()
    run_training_and_evaluation(config)