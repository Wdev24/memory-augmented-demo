# 🧠 Memory-Augmented LLM System

A semantic caching system with intelligent agents that stores and retrieves LLM responses based on query similarity.

## Features

- **Semantic Caching**: Uses sentence-transformers embeddings and cosine similarity (≥0.7 threshold)
- **Three Agents**: Summarization, Planning, and Retrieval with specialized prompts
- **Memory Persistence**: Stores cache across sessions using JSON files
- **Real-time Stats**: Cache hit/miss counters and hit rate tracking
- **Clean UI**: Responsive Flask web interface with agent selection

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   - Get free Hugging Face API key: https://huggingface.co/settings/tokens
   - Add to `.env` file:
     ```
     HF_API_KEY=your_key_here
     ```

3. **Run the app**:
   ```bash
   python app.py
   ```

4. **Access**: http://localhost:5000

## Usage

1. Select an agent (Summarization/Planning/Retrieval)
2. Enter your query
3. Submit and view results with cache status
4. Monitor hit/miss statistics

## Architecture

- `memory.py`: Semantic caching with NumPy-based cosine similarity
- `llm.py`: Hugging Face API integration (Llama 2 + GPT-2 fallback)
- `agent.py`: Agent prompt templates and management
- `app.py`: Flask backend with REST endpoints
- `templates/index.html`: Responsive frontend interface

## Cache Logic

- **Cache Hit** (🟢): Similarity ≥ 0.7 → Return stored response
- **Cache Miss** (🔴): Similarity < 0.7 → Generate new response + store

The system automatically handles model failures with fallback options and persists memory across restarts.