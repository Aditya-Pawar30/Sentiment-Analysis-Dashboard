import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer, AutoModelForSequenceClassification # pyright: ignore[reportMissingImports]
from datetime import datetime

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize sentiment analysis model"""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
    def analyze(self, text):
        """Analyze sentiment of given text"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract scores
        scores = predictions[0].tolist()
        
        # Map to sentiment labels
        sentiment_map = {0: "negative", 1: "positive"}
        sentiment_idx = torch.argmax(predictions).item()
        
        return {
            "text": text,
            "sentiment": sentiment_map[sentiment_idx],
            "confidence": scores[sentiment_idx],
            "scores": {
                "negative": scores[0],
                "positive": scores[1]
            },
            "timestamp": datetime.now()
        }