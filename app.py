from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from memory import SemanticMemory
from llm import generate, test_api_connectivity
import time
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
memory = SemanticMemory(similarity_threshold=0.7)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with improved error handling"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        start_time = time.time()
        
        # Check semantic memory first
        cached_response = memory.search(query)
        
        if cached_response:
            response_time = time.time() - start_time
            print(f"✅ Cache hit for: {query[:50]}...")
            return jsonify({
                'response': cached_response,
                'cached': True,
                'response_time': f"{response_time:.2f}s",
                'stats': memory.get_stats()
            })
        
        # Generate new response
        print(f"🔄 Cache miss - generating new response for: {query[:50]}...")
        llm_response = generate(query)
        
        # Determine if response was generated successfully
        api_success = llm_response and not llm_response.startswith("Error:") and "fallback" not in llm_response.lower()
        
        if api_success:
            # Store successful API responses in memory
            memory.store(query, llm_response)
            print(f"💾 Stored new response in memory")
        else:
            print(f"⚠️ Using fallback response (not cached)")
        
        response_time = time.time() - start_time
        
        return jsonify({
            'response': llm_response,
            'cached': False,
            'api_success': api_success,
            'response_time': f"{response_time:.2f}s",
            'stats': memory.get_stats()
        })
        
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/stats')
def stats():
    """Get memory statistics and API status"""
    try:
        # Test API connectivity
        available_models = test_api_connectivity()
        api_status = len(available_models) > 0
        
        return jsonify({
            **memory.get_stats(),
            'api_status': api_status,
            'available_models_count': len(available_models),
            'sample_models': available_models[:5] if available_models else []
        })
    except Exception as e:
        return jsonify({
            **memory.get_stats(),
            'api_status': False,
            'error': str(e)
        })

@app.route('/clear', methods=['POST'])
def clear_memory():
    """Clear semantic memory"""
    global memory
    try:
        memory = SemanticMemory(similarity_threshold=0.7)
        print("🧹 Memory cleared successfully")
        return jsonify({'message': 'Memory cleared successfully'})
    except Exception as e:
        print(f"❌ Error clearing memory: {e}")
        return jsonify({'error': f'Failed to clear memory: {str(e)}'}), 500

@app.route('/test-api')
def test_api():
    """Test API endpoint for debugging"""
    try:
        test_prompt = "Hello, how are you?"
        start_time = time.time()
        response = generate(test_prompt)
        response_time = time.time() - start_time
        
        return jsonify({
            'test_prompt': test_prompt,
            'response': response,
            'response_time': f"{response_time:.2f}s",
            'api_working': response and not response.lower().startswith('error')
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'api_working': False
        })

if __name__ == '__main__':
    print("🚀 Starting Memory-Augmented LLM Demo...")
    print(f"📊 Initial memory stats: {memory.get_stats()}")
    
    # Check if API key is available
    api_key = os.getenv("TOGETHER_API_KEY", "tgp_v1_5LFzL374MbMoNI6CNLhO5PF7qlosPj8bHazud7LbXJs")
    if api_key:
        print(f"🔑 TogetherAI API key loaded: {api_key[:10]}...")
        
        # Test API connectivity on startup
        print("🔗 Testing API connectivity...")
        available_models = test_api_connectivity()
        if available_models:
            print(f"✅ API is accessible with {len(available_models)} models")
            chat_models = [m for m in available_models if any(keyword in m.lower() for keyword in ['instruct', 'chat', 'turbo'])]
            if chat_models:
                print(f"🎯 Found {len(chat_models)} chat models: {chat_models[:3]}...")
            else:
                print("⚠️ No chat/instruct models found")
        else:
            print("❌ API not accessible, will use fallback responses")
    else:
        print("⚠️ No TogetherAI API key found, will use fallback responses")
    
    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)