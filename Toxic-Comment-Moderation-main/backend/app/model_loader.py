import os
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class to load and manage the model and tokenizer."""

    _instance = None
    _initialized = False
    _model = None
    _tokenizer = None
    _threshold = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelLoader._initialized:
            try:
                self._load_model()
                self._load_threshold()
                ModelLoader._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize ModelLoader: {e}")
                raise

    def _load_model(self):
        """Load model and tokenizer from local final_model directory."""
        try:
            # Get the project root directory (parent of backend/)
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / "final_model"

            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            logger.info(f"Loading model from {model_dir}")

            # Load tokenizer
            self._tokenizer = DistilBertTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True
            )

            # Load model
            self._model = DistilBertForSequenceClassification.from_pretrained(
                str(model_dir),
                local_files_only=True
            )

            # Set device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"Model loaded successfully on device: {self._device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_threshold(self):
        """Load threshold from threshold.json."""
        try:
            project_root = Path(__file__).parent.parent.parent
            threshold_file = project_root / "threshold.json"

            if not threshold_file.exists():
                raise FileNotFoundError(f"Threshold file not found: {threshold_file}")

            with open(threshold_file, 'r') as f:
                threshold_data = json.load(f)

            self._threshold = threshold_data.get("threshold")

            if self._threshold is None:
                raise ValueError("Threshold not found in threshold.json")

            logger.info(f"Threshold loaded: {self._threshold}")

        except Exception as e:
            logger.error(f"Error loading threshold: {e}")
            raise

    def predict(self, text: str) -> Tuple[float, str]:
        """
        Predict toxicity probability and label for given text.

        Args:
            text: Input text to moderate

        Returns:
            tuple: (probability, label) where label is "TOXIC" or "NON-TOXIC"
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        # Tokenize input (max_length=192 as per training)
        inputs = self._tokenizer(
            text,
            max_length=192,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self._device)

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            # Assuming class 1 is toxic (check model config if needed)
            toxic_prob = probabilities[0][1].item()

        # Apply threshold
        label = "TOXIC" if toxic_prob >= self._threshold else "NON-TOXIC"

        return toxic_prob, label

    def predict_batch(self, texts: List[str]) -> Tuple[List[float], List[str]]:
        """
        Predict toxicity for multiple texts.

        Args:
            texts: List of input texts to moderate

        Returns:
            tuple: (probabilities, labels) lists
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        # Tokenize batch
        inputs = self._tokenizer(
            texts,
            max_length=192,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self._device)

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            # Assuming class 1 is toxic - use .tolist() directly instead of numpy
            toxic_probs = probabilities[:, 1].cpu().tolist()

        # Apply threshold
        labels = ["TOXIC" if prob >= self._threshold else "NON-TOXIC" for prob in toxic_probs]

        return toxic_probs, labels

    @property
    def threshold(self) -> float:
        """Get the current threshold value."""
        return self._threshold

    @property
    def device(self) -> str:
        """Get the device being used."""
        return str(self._device)

    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded."""
        return self._model is not None and self._tokenizer is not None and self._threshold is not None

