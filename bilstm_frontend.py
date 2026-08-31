"""Small Streamlit frontend for a BiLSTM sequence tagger.

Run with:
	streamlit run bilstm_frontend.py

The tagger can expose a HTTP endpoint accepting ``{"text": "..."}`` and
returning either ``{"tags": [...]}`` or ``{"tokens": [...], "tags": [...]}``.
"""

import re
from typing import Any

import requests
import importlib
from pathlib import Path
import streamlit as st

TAGGER_ENDPOINT = "http://localhost:8000/predict"
TAGGER_TIMEOUT_SECONDS = 30.0
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def _reset_form() -> None:
	st.session_state.input_text = DEFAULT_TEXT
	st.session_state.pop("uploaded_file", None)
	st.session_state.pop("loaded_file_id", None)
	st.session_state.prediction = None
	st.session_state.selected_sentence_ends = set()
	st.session_state.selection_text = DEFAULT_TEXT


def _tokens(text: str) -> list[str]:
	"""Tokenise text in a predictable way for displaying fallback output."""
	return re.findall(r"\.\.\.|\w+(?:[-']\w+)*|[^\w\s]", text, flags=re.UNICODE)


def _is_sentence_end_token(token: str) -> bool:
	return token in {".", "!", "?", ";", ":", "..."}


def _format_predictions(tokens: list[str], tags: list[str], style: str) -> str:
	if style == "Horizontal":
		return " ".join(f"{token}_{tag}" for token, tag in zip(tokens, tags))
	if style == "Vertical":
		return "\n".join(f"{token}\t{tag}" for token, tag in zip(tokens, tags))
	else:
		return "\n".join(f'<token text="{token}" tag="{tag}" />' for token, tag in zip(tokens, tags))


def _split_into_sentences(tokens: list[str]) -> list[list[str]]:
	"""Split tokenized input at sentence-ending punctuation."""
	sentences = []
	current_sentence = []
	for token in tokens:
		current_sentence.append(token)
		if _is_sentence_end_token(token):
			sentences.append(current_sentence)
			current_sentence = []
	if current_sentence:
		sentences.append(current_sentence)
	return sentences


def _predict(
	endpoint: str,
	text: str,
	selected_sentence_end_indices: list[int],
	timeout: float,
) -> dict[str, Any]:
	# Use the local model loader directly; the server is not required for the current frontend.
	try:
		loader_path = Path(__file__).resolve().parent / "model_loader.py"
		if not loader_path.exists():
			# try one level up
			loader_path = Path(__file__).resolve().parent.parent / "Code" / "model_loader.py"
		if not loader_path.exists():
			raise requests.RequestException("Local model_loader not found")

		spec = importlib.util.spec_from_file_location("local_model_loader", loader_path)
		if spec is None or spec.loader is None:
			raise requests.RequestException("Could not load local model_loader")
		model_loader = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(model_loader)

		# try checkpoint in workspace root
		ckpt = Path(__file__).resolve().parent.parent / "bilstm_pos_tagger_bnc.pth"
		if not ckpt.exists():
			raise requests.RequestException("Model checkpoint not found")

		model, idx2tag = model_loader.load_checkpoint(str(ckpt))
		all_tokens = []
		all_tags = []
		for sentence_tokens in _split_into_sentences(_tokens(text)):
			tags, used_tokens = model_loader.predict_sentence(
				model, idx2tag, sentence_tokens
			)
			if len(used_tokens) != len(tags):
				raise ValueError(
					"The model returned a different number of tokens and tags for one sentence."
				)
			all_tokens.extend(used_tokens)
			all_tags.extend(tags)
		return {"tokens": all_tokens, "tags": all_tags}

	except requests.RequestException:
		raise
	except RuntimeError as exc:
		raise requests.RequestException(
			f"The model checkpoint is corrupt or incomplete. "
			f"Please retrain the model and save a fresh checkpoint before tagging. Details: {exc}"
		) from exc
	except Exception as exc:
		raise requests.RequestException(f"Local tagger failed: {exc}")


def main() -> None:
	st.set_page_config(page_title="BiLSTM Tagger", page_icon="🏷️", layout="wide")
	st.title("Welcome to our FREE BiLSTM part-of-speech tagger!")
	st.caption("Enter text below and or upload a file.")
	st.markdown(
		"""
		This tool allows you to input text and receive predicted tags for each token in the text.

		You can use one of the following methods to provide input text for tagging:

		- **Direct Input:** Type or paste your text directly into the input box below.
		- **File Upload:** Upload a plain text file (.txt) containing the text you want to tag.

		The tagger will return a list of predicted tags for each token in the input text.
		Make sure to only upload plain text files (.txt) to avoid any issues with file parsing.
		Please note that the tagger may not always produce accurate results, and it is recommended to review the predictions carefully.
		"""
	)

	result_style = st.radio(
		"**Result style**",
		options=["Horizontal", "Vertical", "Pseudo-XML"],
		horizontal=True,
		key="result_style",
	)

	uploaded_file = st.file_uploader(
		"**Upload a .txt file**",
		accept_multiple_files=False,
		type=["txt"],
		help="Drag and drop a plain text file here, or click to browse.",
		key="uploaded_file",
	)
	uploaded_text = None
	if uploaded_file is not None:
		raw_text = uploaded_file.getvalue()
		try:
			uploaded_text = raw_text.decode("utf-8")
		except UnicodeDecodeError:
			uploaded_text = raw_text.decode("latin-1")
		file_id = (uploaded_file.name, uploaded_file.size, hash(raw_text))
		if st.session_state.get("loaded_file_id") != file_id:
			st.session_state.input_text = uploaded_text
			st.session_state.loaded_file_id = file_id

	text = st.text_area(
		"**Input text**",
		value=DEFAULT_TEXT,
		height=150,
		placeholder="Type a sentence to tag...",
		key="input_text",
	)

	if "prediction" not in st.session_state:
		st.session_state.prediction = None
	if "selected_sentence_ends" not in st.session_state:
		st.session_state.selected_sentence_ends = set()
	input_tokens = _tokens(text)
	if st.session_state.get("selection_text") != text:
		st.session_state.selection_text = text
		st.session_state.selected_sentence_ends = {
			index
			for index, token in enumerate(input_tokens)
			if _is_sentence_end_token(token)
		}

	tag_column, reset_column = st.columns(2)
	with tag_column:
		tag_clicked = st.button("Tag text", type="primary", use_container_width=True)
	with reset_column:
		st.button("Reset form", use_container_width=True, on_click=_reset_form)

	if tag_clicked:
		if not text.strip():
			st.warning("Please enter some text first.")
		else:
			try:
				result = _predict(
					TAGGER_ENDPOINT,
					text,
					sorted(st.session_state.selected_sentence_ends),
					TAGGER_TIMEOUT_SECONDS,
				)
				tokens, tags = result.get("tokens"), result.get("tags")
				if len(tokens) != len(tags):
					raise ValueError("The number of tokens and tags does not match.")
				st.session_state.prediction = (tokens, tags)
			except requests.RequestException as exc:
				st.error(str(exc))
			except (ValueError, TypeError) as exc:
				st.error(f"Invalid tagger response: {exc}")

	if st.session_state.prediction is not None:
		tokens, tags = st.session_state.prediction
		st.subheader("Predictions")
		st.code(_format_predictions(tokens, tags, result_style), language="xml" if result_style == "pseudo-xml" else "text")


if __name__ == "__main__":
	main()
