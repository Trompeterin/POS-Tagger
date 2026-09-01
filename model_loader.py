from pathlib import Path
import json
import re

import torch
from transformers import BertModel, BertTokenizerFast


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertBiLSTMTagger(torch.nn.Module):
    """Minimal BiLSTM + BERT POS tagger used by the standalone loader."""

    def __init__(self, tagset_size, model_name="bert-base-multilingual-cased", hidden_dim=64, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        if not freeze_bert:
            for parameter in list(self.bert.parameters())[-4:]:
                parameter.requires_grad = True

        self.lstm = torch.nn.LSTM(
            self.bert.config.hidden_size,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out, _ = self.lstm(bert_out.last_hidden_state)
        return self.fc(out)


def encode_sentence(sent, tokenizer, tag2idx, max_seq_length=128):
    """Encode a sentence into token IDs, attention mask, and label IDs."""
    if tokenizer is None:
        raise RuntimeError("Tokenizer must be provided.")
    if tag2idx is None:
        tag2idx = {"<UNK>": 0}

    words = [word for word, _ in sent]
    tags = [tag for _, tag in sent]

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
    )
    word_ids = encoding.word_ids()

    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-1)
        elif word_idx != previous_word_idx:
            label_ids.append(tag2idx.get(tags[word_idx], tag2idx.get("<UNK>", 0)))
        else:
            label_ids.append(-1)
        previous_word_idx = word_idx

    return encoding["input_ids"], encoding["attention_mask"], label_ids


def preprocess_sentence(sentence):
    """Split common contractions into tokens that keep the tagger word-level alignment stable."""
    if sentence is None:
        return []
    if isinstance(sentence, str):
        sentence = sentence.split()

    processed_tokens = []
    for token in sentence:
        token = str(token)
        if re.match(r"^\w+n't$", token):
            processed_tokens.append(token[:-3])
            processed_tokens.append("n't")
        elif re.match(r"^\w+'\w+$", token):
            processed_tokens.append(token[:-2])
            processed_tokens.append(token[-2:])
        else:
            processed_tokens.append(token)
    return processed_tokens


def inference(model, sentence, tokenizer, tag2idx, idx2tag=None, device=None, max_seq_length=128):
    """Run one sentence through the model and return predicted tags plus tokens."""
    if idx2tag is None:
        idx2tag = {v: k for k, v in tag2idx.items()}

    model_device = device if device is not None else next(model.parameters()).device
    model.to(model_device)
    model.eval()

    with torch.no_grad():
        words = preprocess_sentence(sentence)
        sent = [(word, "<UNK>") for word in words]
        input_ids, attn_mask, label_ids = encode_sentence(sent, tokenizer, tag2idx, max_seq_length)
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids_t = torch.tensor([input_ids], device=model_device)
        attn_mask_t = torch.tensor([attn_mask], device=model_device)

        logits = model(input_ids_t, attn_mask_t)
        preds = torch.argmax(logits, dim=-1)[0]

        word_ids = encoding.word_ids()
        predicted_tags = []
        tokens = []
        previous_word_id = None

        for position, pred in enumerate(preds):
            if label_ids[position] == -1:
                continue

            word_id = word_ids[position]
            if word_id is None or word_id == previous_word_id:
                continue

            previous_word_id = word_id
            token = words[word_id]
            if not token:
                continue

            tokens.append(token)
            predicted_tags.append(idx2tag.get(pred.item(), "<UNK>"))

    return predicted_tags, tokens


def load_checkpoint(path: str, device: torch.device | None = None):
    if device is None:
        device = DEFAULT_DEVICE

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(path, map_location=device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint '{path}' is corrupted or incomplete. "
            "This usually happens when the save was interrupted or the file was truncated. "
            "Recreate the checkpoint by running the training script again and saving the model cleanly."
        ) from exc

    if isinstance(ckpt, dict):
        model_state = ckpt.get("model_state")
        if model_state is None:
            model_state = ckpt.get("model_state_dict")

        tag2idx = ckpt.get("tag2idx")
        idx2tag = ckpt.get("idx2tag")

        if tag2idx is None and idx2tag is not None:
            tag2idx = {tag: idx for idx, tag in idx2tag.items()}

        if model_state is not None and tag2idx is not None:
            pass
        elif model_state is not None:
            raise RuntimeError(
                f"Checkpoint '{path}' contains a model state but no tag mapping. "
                "Expected keys: 'tag2idx' or 'idx2tag'."
            )
        else:
            model_state = ckpt
            tag2idx = None
            for meta_name in (p.with_suffix(".meta.json"), p.with_suffix(".json")):
                if meta_name.exists():
                    try:
                        data = json.loads(meta_name.read_text(encoding="utf-8"))
                        tag2idx = data.get("tag2idx")
                        if tag2idx:
                            break
                    except Exception:
                        pass

            if tag2idx is None:
                raise RuntimeError(
                    f"Checkpoint '{path}' does not contain a tag mapping and no fallback was found."
                    " Re-save your checkpoint as {'model_state_dict': model.state_dict(), 'tag2idx': tag2idx}"
                    " or {'model_state': model.state_dict(), 'tag2idx': tag2idx}."
                )
    else:
        model_state = ckpt
        tag2idx = None
        raise RuntimeError(
            f"Checkpoint '{path}' is a raw state_dict but no tag mapping was found. "
            "Please re-save the checkpoint with a tag mapping."
        )

    idx2tag = {v: k for k, v in tag2idx.items()}
    model = BertBiLSTMTagger(
        len(tag2idx),
        model_name="bert-base-multilingual-cased",
        hidden_dim=64,
        freeze_bert=True,
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, idx2tag


def predict_sentence(model, idx2tag, sentence_tokens):
    tag2idx = {tag: idx for idx, tag in idx2tag.items()}
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    device = next(model.parameters()).device
    return inference(model, sentence_tokens, tokenizer, tag2idx, idx2tag=idx2tag, device=device)
