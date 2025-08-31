import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

def analyze_sentiment(text):
    if not text or not HF_API_KEY:
        return "NEUTRAL", 0.0
    url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        response = requests.post(url, headers=headers, json={"inputs": text})
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            return result[0][0]["label"], result[0][0]["score"]
        elif isinstance(result, list) and "label" in result[0]:
            return result[0]["label"], result[0]["score"]
        else:
            return "NEUTRAL", 0.0
    except Exception as e:
        print("Error:", e)
        return "NEUTRAL", 0.0

print(analyze_sentiment("I feel great and full of energy!"))
print(analyze_sentiment("I am so stressed and exhausted."))
