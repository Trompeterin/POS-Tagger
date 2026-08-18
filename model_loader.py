import importlib.util
from pathlib import Path
import torch

# Load the bnc_included_bilstm module from the POS-Tagger folder
module_path = Path(__file__).resolve().parent.parent / "bnc_included_bilstm.py"
spec = importlib.util.spec_from_file_location("bnc_included_bilstm", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load model module from {module_path}")

bnc_included_bilstm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bnc_included_bilstm)


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # Supported checkpoint formats:
    # 1) {'model_state_dict': ..., 'tag2idx': ..., 'idx2tag': ...}
    # 2) {'model_state': ..., 'tag2idx': ...}
    # 3) raw state_dict
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
            tag2idx = getattr(bnc_included_bilstm, "tag2idx", None)
            if tag2idx is None:
                import json

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
        tag2idx = getattr(bnc_included_bilstm, "tag2idx", None)
        if tag2idx is None:
            raise RuntimeError(
                f"Checkpoint '{path}' is a raw state_dict but no fallback tag mapping was found. "
                "Please re-save the checkpoint with a tag mapping."
            )

    idx2tag = {v: k for k, v in tag2idx.items()}
    model = bnc_included_bilstm.BertBiLSTMTagger(
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
    # build tag2idx
    tag2idx = {tag: idx for idx, tag in idx2tag.items()}
    try:
        from transformers import BertTokenizerFast

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the BERT tokenizer for inference. "
            "Make sure the transformers package and model files are available."
        ) from exc

    device = next(model.parameters()).device
    return bnc_included_bilstm.inference(model, sentence_tokens, tokenizer, tag2idx, idx2tag=idx2tag, device=device)
