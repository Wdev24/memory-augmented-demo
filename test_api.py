#!/usr/bin/env python3
"""
Simple script to test TogetherAI API connectivity
Run this to debug API issues: python test_api.py
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_together_api():
    """Test TogetherAI API with your key"""
    
    # Your API key
    api_key = os.getenv("TOGETHER_API_KEY", "tgp_v1_5LFzL374MbMoNI6CNLhO5PF7qlosPj8bHazud7LbXJs")
    
    print("🧪 Testing TogetherAI API...")
    print(f"🔑 Using API key: {api_key[:15]}...{api_key[-5:]}")
    print("-" * 50)
    
    # Test 1: Check if API key is valid by listing models
    print("📋 Test 1: Listing available models...")
    try:
        models_url = "https://api.together.xyz/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(models_url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            models_data = response.json()
            print("✅ API key is valid!")
            
            # Handle both list and dict formats
            if isinstance(models_data, list):
                models_list = models_data
                print(f"📊 Found {len(models_list)} models")
            else:
                models_list = models_data.get('data', [])
                print(f"📊 Found {len(models_list)} models")
            
            # Show first few models
            for i, model in enumerate(models_list[:10]):
                if isinstance(model, str):
                    print(f"  {i+1}. {model}")
                else:
                    print(f"  {i+1}. {model.get('id', model.get('name', 'unknown'))}")
        else:
            print("❌ API key validation failed")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False
    
    print("-" * 50)
    
    # Test 2: Try a simple chat completion
    print("💬 Test 2: Simple chat completion...")
    
    # Models to try in order
    test_models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1", 
        "NousResearch/Nous-Hermes-2-Yi-34B",
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
    ]
    
    chat_url = "https://api.together.xyz/v1/chat/completions"
    
    for model in test_models:
        print(f"\n🤖 Testing model: {model}")
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "What is electricity? Please give a brief explanation."}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(chat_url, headers=headers, json=payload, timeout=15)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0]["message"]["content"]
                    print("✅ SUCCESS!")
                    print(f"🤖 Response: {answer[:200]}...")
                    return True
                else:
                    print("❌ No choices in response")
                    print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print("❌ Request failed")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS:")
    print("All models failed. Possible issues:")
    print("1. API key might be invalid or expired")
    print("2. Account might need credits/billing setup")
    print("3. Network connectivity issues")
    print("4. TogetherAI service might be down")
    print("\n💡 Try:")
    print("- Check your TogetherAI account dashboard")
    print("- Verify billing/credits are set up")
    print("- Try a different API key")
    
    return False

if __name__ == "__main__":
    test_together_api()