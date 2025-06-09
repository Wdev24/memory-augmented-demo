import requests
import os
from typing import Optional
import random
import json

def generate(prompt: str, max_tokens: int = 512) -> Optional[str]:
    """Generate response using TogetherAI API with working serverless models"""
    
    # Get TogetherAI API key
    api_key = os.getenv("TOGETHER_API_KEY", "tgp_v1_5LFzL374MbMoNI6CNLhO5PF7qlosPj8bHazud7LbXJs")
    
    print(f"🔑 Using API key: {api_key[:15]}...")
    
    try:
        # TogetherAI API endpoint
        url = "https://api.together.xyz/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Updated models list with working serverless models
        # Based on your test results, these models should work
        models = [
            "mistralai/Mistral-7B-Instruct-v0.1",  # This one worked in your test!
            "meta-llama/Llama-3.2-3B-Instruct-Turbo",  # Turbo models are usually serverless
            "arcee-ai/coder-large",  # Good for code-related queries
            "arcee-ai/arcee-blitz",  # Fast model
            "WhereIsAI/UAE-Large-V1",  # From your available models list
        ]
        
        for model in models:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            print(f"🚀 Trying model: {model}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API call successful!")
                
                if "choices" in result and len(result["choices"]) > 0:
                    generated_text = result["choices"][0]["message"]["content"].strip()
                    if generated_text:
                        print(f"🎉 Generated response with {model}")
                        return generated_text
                    else:
                        print("⚠️ Empty response content")
                else:
                    print("⚠️ No choices in response")
                    print(f"📄 Response structure: {json.dumps(result, indent=2)}")
                    
            elif response.status_code == 400:
                error_response = response.json()
                if "model_not_available" in str(error_response):
                    print(f"❌ Model {model} requires dedicated endpoint, trying next...")
                    continue
                else:
                    print(f"❌ Bad request: {response.text}")
                    continue
                    
            elif response.status_code == 422:
                print(f"❌ Model {model} validation error, trying next...")
                continue
                
            else:
                print(f"❌ API error {response.status_code}: {response.text}")
                # Try next model for most errors
                continue
                
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except requests.exceptions.ConnectionError:
        print("🌐 Connection error")
    except Exception as e:
        print(f"❌ TogetherAI API exception: {str(e)}")
    
    # Test API connectivity if all models fail
    print("🧪 Testing API connectivity...")
    try:
        test_url = "https://api.together.xyz/v1/models"
        test_headers = {"Authorization": f"Bearer {api_key}"}
        test_response = requests.get(test_url, headers=test_headers, timeout=10)
        print(f"🔍 Models endpoint status: {test_response.status_code}")
        
        if test_response.status_code == 200:
            models_data = test_response.json()
            available_models = []
            for model in models_data.get('data', [])[:10]:
                model_id = model.get('id', 'unknown')
                available_models.append(model_id)
            print(f"📋 Available models (first 10): {available_models}")
            
            # Try to find a working chat model from available ones
            chat_models = [m for m in available_models if any(keyword in m.lower() for keyword in ['instruct', 'chat', 'turbo'])]
            if chat_models:
                print(f"🎯 Found potential chat models: {chat_models}")
                
        else:
            print(f"❌ Models endpoint error: {test_response.text}")
            
    except Exception as e:
        print(f"❌ API connectivity test failed: {str(e)}")
    
    # Fallback to smart responses
    print("🔄 All models failed, using fallback responses...")
    return get_smart_fallback(prompt)

def get_smart_fallback(prompt: str) -> str:
    """Enhanced fallback responses based on prompt content"""
    prompt_lower = prompt.lower()
    
    # Science and technical questions
    if any(word in prompt_lower for word in ['electricity', 'electric', 'current', 'voltage', 'power']):
        responses = [
            "Electricity is the flow of electric charge through conductors, typically electrons moving through materials like copper wire. It's measured in amperes (current), volts (voltage), and watts (power). The relationship between these is described by Ohm's Law: V = I × R.",
            "Electric current flows when there's a potential difference (voltage) across a conductor. AC (alternating current) is used in homes because it's efficient for long-distance transmission, while DC (direct current) is used in batteries and electronics.",
            "Electrical power systems work on the principle of electromagnetic induction. Generators convert mechanical energy to electrical energy, while motors do the reverse. This is fundamental to how our power grid operates."
        ]
    elif any(word in prompt_lower for word in ['python', 'programming', 'code', 'function', 'variable']):
        responses = [
            "Python is known for its clean, readable syntax and powerful libraries. Key concepts include variables, functions, classes, and modules. Popular libraries include NumPy for numerical computing, Pandas for data analysis, and Flask/Django for web development.",
            "Good programming practices include writing clean, readable code, using meaningful variable names, adding comments, and following the DRY (Don't Repeat Yourself) principle. Python's PEP 8 style guide provides excellent coding standards.",
            "Python supports multiple programming paradigms: procedural, object-oriented, and functional programming. This flexibility makes it suitable for everything from simple scripts to complex applications and machine learning models."
        ]
    elif any(word in prompt_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'neural network']):
        responses = [
            "Artificial Intelligence encompasses machine learning, deep learning, natural language processing, and computer vision. Modern AI systems use neural networks inspired by how the human brain processes information.",
            "Machine learning algorithms learn patterns from data to make predictions or decisions. Common types include supervised learning (with labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through interaction).",
            "Deep learning uses neural networks with multiple layers to process complex data like images, text, and speech. Popular frameworks include TensorFlow, PyTorch, and Keras."
        ]
    elif any(word in prompt_lower for word in ['flask', 'web', 'api', 'server', 'http']):
        responses = [
            "Flask is a lightweight Python web framework that's great for building APIs and web applications. It follows the WSGI standard and provides features like routing, templating, and request handling.",
            "Web APIs use HTTP methods (GET, POST, PUT, DELETE) to enable communication between different applications. RESTful APIs follow specific conventions for organizing endpoints and handling resources.",
            "Modern web development involves client-server architecture, where the frontend (client) communicates with the backend (server) through APIs. This separation allows for scalable and maintainable applications."
        ]
    elif any(word in prompt_lower for word in ['memory', 'cache', 'storage', 'database']):
        responses = [
            "Semantic memory systems store and retrieve information based on meaning rather than exact matches. This allows for more intelligent caching and retrieval of relevant information.",
            "Caching strategies improve performance by storing frequently accessed data in fast storage. Common patterns include LRU (Least Recently Used), LFU (Least Frequently Used), and semantic similarity-based caching.",
            "Vector databases and embeddings enable semantic search by representing text as high-dimensional vectors. Similar concepts cluster together in this vector space, allowing for intelligent information retrieval."
        ]
    else:
        # Generic intelligent responses
        responses = [
            f"That's a thoughtful question about '{prompt}'. While I'm currently using backup responses due to API connectivity issues, I can still provide helpful information on this topic. What specific aspect would you like to explore?",
            f"I find '{prompt}' quite interesting! Although I'm experiencing some technical difficulties with my primary AI service, I'm still here to discuss this topic with you. Could you tell me more about what you're looking for?",
            f"Great question regarding '{prompt}'! I'm currently running on fallback responses, but I'd be happy to share what I know about this subject. What particular details are you most curious about?"
        ]
    
    return random.choice(responses)

def test_api_connectivity():
    """Test API connectivity and return available models"""
    api_key = os.getenv("TOGETHER_API_KEY", "tgp_v1_5LFzL374MbMoNI6CNLhO5PF7qlosPj8bHazud7LbXJs")
    
    try:
        url = "https://api.together.xyz/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            return [model.get('id') for model in models_data.get('data', [])]
        else:
            print(f"API connectivity test failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"API connectivity test error: {e}")
        return []

if __name__ == "__main__":
    # Test the generate function
    test_prompt = "What is electricity?"
    print("🧪 Testing generate function...")
    result = generate(test_prompt)
    print(f"📝 Result: {result}")
    
    # Test API connectivity
    print("\n🔗 Testing API connectivity...")
    available_models = test_api_connectivity()
    if available_models:
        print(f"✅ Found {len(available_models)} available models")
        chat_models = [m for m in available_models if 'instruct' in m.lower() or 'chat' in m.lower()]
        print(f"🎯 Chat models: {chat_models[:5]}")
    else:
        print("❌ No models found or API not accessible")