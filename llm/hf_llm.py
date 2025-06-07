# llm/hf_llm.py

import os
import requests
from dotenv import load_dotenv

# Load the .env file
load_dotenv()  
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/"

def generate_with_hf(model_name: str, prompt: str, max_length: int = 64, temperature: float = 0.7):
    """
    1) Read HUGGINGFACEHUB_API_TOKEN from environment.
    2) POST to Hugging Face Inference API.
    3) Return the 'generated_text' from JSON response.
    """
    if HF_API_KEY is None:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not set in .env")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "return_full_text": False
        }
    }

    response = requests.post(HF_API_URL + model_name, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Usually returns a list of dicts with 'generated_text'
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    else:
        return str(data)
