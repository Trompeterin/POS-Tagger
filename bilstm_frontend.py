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
import zipfile
import io
from spellchecker import SpellChecker
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


TAGGER_ENDPOINT = "http://localhost:8000/predict"
TAGGER_TIMEOUT_SECONDS = 30.0
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
MAX_SINGLE_UPLOAD_MB = 5
MAX_ZIP_UPLOAD_MB = 10
MAX_ZIP_FILES = 20
MAX_ZIP_FILE_BYTES = 5 * 1024 * 1024
MAX_ZIP_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_TEXT_CHARS = 500_000


def _decode_uploaded_text(raw_content: bytes) -> str:
	try:
		return raw_content.decode("utf-8")
	except UnicodeDecodeError:
		return raw_content.decode("latin-1")


def _read_zip_text(uploaded_file: Any) -> str:
	if uploaded_file.size > MAX_ZIP_UPLOAD_MB * 1024 * 1024:
		raise ValueError(f"ZIP uploads must be no larger than {MAX_ZIP_UPLOAD_MB} MB.")

	with zipfile.ZipFile(uploaded_file) as archive:
		content_files = []
		total_uncompressed_bytes = 0
		for file_info in archive.infolist():
			path_parts = file_info.filename.replace("\\", "/").split("/")
			if file_info.is_dir() or "__MACOSX" in path_parts or any(
				part.startswith("._") for part in path_parts
			):
				continue
			if not file_info.filename.lower().endswith((".txt", ".csv")):
				continue

			content_files.append(file_info)
			if len(content_files) > MAX_ZIP_FILES:
				raise ValueError(f"ZIP archives may contain at most {MAX_ZIP_FILES} text or CSV files.")
			if file_info.file_size > MAX_ZIP_FILE_BYTES:
				raise ValueError("Each file inside a ZIP archive must be no larger than 5 MB uncompressed.")
			total_uncompressed_bytes += file_info.file_size
			if total_uncompressed_bytes > MAX_ZIP_UNCOMPRESSED_BYTES:
				raise ValueError("The total uncompressed text and CSV data in a ZIP archive must not exceed 25 MB.")

		if not content_files:
			raise ValueError("The ZIP archive contains no .txt or .csv files to process.")

		texts = []
		total_chars = 0
		actual_uncompressed_bytes = 0
		for file_info in content_files:
			with archive.open(file_info) as content_file:
				raw_content = content_file.read(MAX_ZIP_FILE_BYTES + 1)
			if len(raw_content) > MAX_ZIP_FILE_BYTES:
				raise ValueError("A file inside the ZIP archive exceeds the 5 MB uncompressed limit.")
			actual_uncompressed_bytes += len(raw_content)
			if actual_uncompressed_bytes > MAX_ZIP_UNCOMPRESSED_BYTES:
				raise ValueError("The total uncompressed text and CSV data in a ZIP archive must not exceed 25 MB.")

			file_text = _decode_uploaded_text(raw_content)
			total_chars += len(file_text) + 1
			if total_chars > MAX_UPLOAD_TEXT_CHARS:
				raise ValueError(f"Uploaded text must not exceed {MAX_UPLOAD_TEXT_CHARS:,} characters.")
			texts.append(file_text)

	return "\n".join(texts)


def _reset_form() -> None:
	st.session_state.input_text = DEFAULT_TEXT
	st.session_state.pop("uploaded_file", None)
	st.session_state.pop("loaded_file_id", None)
	st.session_state.prediction = None
	st.session_state.selected_sentence_ends = set()
	st.session_state.selection_text = DEFAULT_TEXT

def _spell_check() -> None:
	if "input_text" in st.session_state:
		text = st.session_state.input_text
		try:
			spell = SpellChecker()
			spell_german = SpellChecker(language='de')
			# split text into single words and split punctuation from words as we want to keep them
			for word in text.split():
				if not word.isalpha():
					# split punctuation from word
					# e.g. "word." -> "word" and "."
					word_parts = re.findall(r"\w+|[^\w\s]", word, flags=re.UNICODE)
					for part in word_parts:
						if spell.unknown([part]):
							try:
								language = detect(part)
							except LangDetectException:
								return None
							if language == 'de':
								corrected_word = spell_german.correction(part)
							else:
								corrected_word = spell.correction(part)
							if corrected_word != part and corrected_word is not None:
								text = text.replace(part, corrected_word)
				elif spell.unknown([word]):
					try:
						language = detect(word)
					except LangDetectException:
						return None
					if language == 'de':
						corrected_word = spell_german.correction(word)
					else:
						corrected_word = spell.correction(word)
					if corrected_word != word and corrected_word is not None:
						text = text.replace(word, corrected_word)
			corrected_text = text
			st.session_state.input_text = corrected_text
		except Exception as e:
			st.error(f"An error occurred during spell check: {e}")
	else:
		st.warning("No input text found to spell check.")


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

		# resolve checkpoint in multiple realistic locations
		candidate_paths = [
			Path(__file__).resolve().parent.parent / "bilstm_pos_tagger_bnc.pth",
			Path(__file__).resolve().parent / "bilstm_pos_tagger_bnc.pth",
			Path(__file__).resolve().parent.parent / "POS-Tagger" / "bilstm_pos_tagger_bnc.pth",
		]
		ckpt = next((p for p in candidate_paths if p.exists()), None)
		if ckpt is None:
			raise requests.RequestException(
				"Model checkpoint not found. Expected one of: "
				+ ", ".join(str(p) for p in candidate_paths)
			)

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
	st.title("Welcome to our free BiLSTM part-of-speech tagger!")
	st.caption("Enter text below and or upload a file.")
	st.markdown(
		"""
		This tool allows you to input text and receive predicted tags for each token in the text. The tagset used is the [UCREL CLAWS5 Tagset](https://ucrel.lancs.ac.uk/claws5tags.html).
		This tagger was trained specifically for written German Learner English.

		A BiLSTM (Bidirectional Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture that is commonly used for sequence tagging tasks, such as part-of-speech tagging. 
		It processes the input sequence in both forward and backward directions, allowing it to capture context from both past and future tokens. 
		This bidirectional processing helps improve the accuracy of predictions by considering the surrounding context of each token in the sequence.

		You can use one of the following methods to provide input text for tagging:

		- **Direct Input:** Type or paste your text directly into the input box below.
		- **File Upload:** Upload a plain text file (.txt) or CSV file (.csv) containing the text you want to tag.
		- **Bulk Upload:** Upload multiple files (.txt or .csv but not both) as a zip archive. 

		The tagger will return a list of predicted tags for the tokens in the input text.
		Make sure to only upload plain text files (.txt) to avoid any issues with file parsing.
		Please note that the tagger may not always produce accurate results, and it is recommended to review the predictions carefully.

		Sentences longer than 100 Words may be truncated, and the tagger may not handle them correctly. Please keep this in mind when providing input text.
		"""
	)

	result_style = st.radio(
		"**Result style**",
		options=["Horizontal", "Vertical", "Pseudo-XML"],
		horizontal=True,
		key="result_style",
	)

	uploaded_file = st.file_uploader(
		"**Upload a .txt or .csv file**",
		accept_multiple_files=False,
		type=["txt", "csv"],
		max_upload_size=MAX_SINGLE_UPLOAD_MB,
		help="Drag and drop a plain text file or CSV file here, or click to browse.",
		key="uploaded_file",
	)

	uploaded_bulk = st.file_uploader(
		"**Upload a zip archive**",
		accept_multiple_files=False,
		type=["zip"],
		max_upload_size=MAX_ZIP_UPLOAD_MB,
		help="Drag and drop a zip archive containing .txt files or .csv files here, or click to browse.",
		key="uploaded_bulk",
	)
	
	uploaded_text = None
	if uploaded_file is not None:
		try:
			if uploaded_file.size > MAX_SINGLE_UPLOAD_MB * 1024 * 1024:
				raise ValueError(f"Text and CSV uploads must be no larger than {MAX_SINGLE_UPLOAD_MB} MB.")
			if uploaded_file.type in ["text/plain", "text/csv"]:
				raw_text = uploaded_file.getvalue()
				uploaded_text = _decode_uploaded_text(raw_text)
				if uploaded_file.type == "text/csv":
					uploaded_text = "\n".join(uploaded_text.splitlines())
				if len(uploaded_text) > MAX_UPLOAD_TEXT_CHARS:
					raise ValueError(f"Uploaded text must not exceed {MAX_UPLOAD_TEXT_CHARS:,} characters.")
				file_id = (uploaded_file.name, uploaded_file.size, hash(uploaded_text))
				if st.session_state.get("loaded_file_id") != file_id:
					st.session_state.input_text = uploaded_text
					st.session_state.loaded_file_id = file_id
		except ValueError as exc:
			st.error(str(exc))
	elif uploaded_bulk is not None:
		try:
			uploaded_text = _read_zip_text(uploaded_bulk)
			file_id = (uploaded_bulk.name, uploaded_bulk.size, hash(uploaded_text))
			if st.session_state.get("loaded_file_id") != file_id:
				st.session_state.input_text = uploaded_text
				st.session_state.loaded_file_id = file_id
		except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as exc:
			st.error(f"Could not process ZIP upload: {exc}")

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

	tag_column, reset_column, spell_check_column = st.columns(3)
	with tag_column:
		tag_clicked = st.button("Tag text", type="primary", use_container_width=True)
	with reset_column:
		st.button("Reset form", use_container_width=True, on_click=_reset_form)
	with spell_check_column:
		st.button("Spell check", use_container_width=True, on_click=_spell_check)

	if tag_clicked:
		if not text.strip():
			st.warning("Please enter some text first.")
		else:
			with st.spinner("Please be patient while the model generates the tags..."):
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

	# button to download the predictions as a text file
	if st.session_state.prediction is not None:
		tokens, tags = st.session_state.prediction
		predictions_text = _format_predictions(tokens, tags, result_style)
		st.download_button(
			label="Download predictions as text file",
			data=predictions_text,
			file_name="predictions.txt",
			mime="text/plain",
		)
	# button to download the predictions as a CSV file
	if st.session_state.prediction is not None:
		tokens, tags = st.session_state.prediction
		# make sure that commas in tokens are handled correctly by enclosing the token in double quotes if it contains a comma
		predictions_csv = "Token,Tag\n" + "\n".join(f'"{token.replace("\"", "\"\"")}",{tag}' for token, tag in zip(tokens, tags))
		st.download_button(
			label="Download predictions as CSV file",
			data=predictions_csv,
			file_name="predictions.csv",
			mime="text/csv",
		)
	# button that should only be visible if the user uploaded a zip file, to download the predictions as a zip file in either CSV or TXT format 
	if uploaded_bulk is not None and st.session_state.prediction is not None:
		tokens, tags = st.session_state.prediction
		predictions_csv = "Token,Tag\n" + "\n".join(f'"{token.replace("\"", "\"\"")}",{tag}' for token, tag in zip(tokens, tags))
		zip_buffer = io.BytesIO()
		with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
			zip_file.writestr("predictions.csv", predictions_csv)
		st.download_button(
			label="Download predictions as ZIP file containing CSV",
			data=zip_buffer.getvalue(),
			file_name="predictions.zip",
			mime="application/zip",
		)
	if uploaded_bulk is not None and st.session_state.prediction is not None:
			tokens, tags = st.session_state.prediction
			predictions_csv = "Token,Tag\n" + "\n".join(f"{token},{tag}" for token, tag in zip(tokens, tags))
			zip_buffer = io.BytesIO()
			with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
				zip_file.writestr("predictions.txt", _format_predictions(tokens, tags, result_style))
			st.download_button(
				label="Download predictions as ZIP file containing TXT",
				data=zip_buffer.getvalue(),
				file_name="predictions.zip",
				mime="application/zip",
			)


if __name__ == "__main__":
	main()
