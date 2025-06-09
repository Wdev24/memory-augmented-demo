from abc import ABC, abstractmethod
from typing import Dict, Any
from llm import LLMClient, MockLLM
from memory import MemoryCache

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, llm_client, memory_cache: MemoryCache):
        self.llm_client = llm_client
        self.memory_cache = memory_cache
        self.name = self.__class__.__name__
    
    @abstractmethod
    def get_prompt_prefix(self) -> str:
        """Get the prompt prefix for this agent"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of what this agent does"""
        pass
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query with caching"""
        # Create the full prompt with agent prefix
        full_prompt = f"{self.get_prompt_prefix()} {user_query}"
        
        # Check cache first
        cached_response, is_hit, similarity = self.memory_cache.get_cached_response(full_prompt)
        
        if is_hit:
            return {
                'response': cached_response,
                'is_cache_hit': True,
                'similarity_score': similarity,
                'agent': self.name,
                'model_used': None
            }
        
        # Cache miss - generate new response
        try:
            response = self.llm_client.generate_response(full_prompt)
            
            # Store in cache
            self.memory_cache.add_to_cache(full_prompt, response)
            
            return {
                'response': response,
                'is_cache_hit': False,
                'similarity_score': 0.0,
                'agent': self.name,
                'model_used': self.llm_client.get_current_model()
            }
            
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'is_cache_hit': False,
                'similarity_score': 0.0,
                'agent': self.name,
                'model_used': None,
                'error': True
            }

class SummarizationAgent(BaseAgent):
    """Agent specialized in summarizing content"""
    
    def get_prompt_prefix(self) -> str:
        return "Summarize this:"
    
    def get_description(self) -> str:
        return "Summarizes long text into key points and main ideas"

class PlanningAgent(BaseAgent):
    """Agent specialized in creating plans and strategies"""
    
    def get_prompt_prefix(self) -> str:
        return "Plan the following task:"
    
    def get_description(self) -> str:
        return "Creates step-by-step plans and strategies for tasks and projects"

class RetrievalAgent(BaseAgent):
    """Agent specialized in extracting key information"""
    
    def get_prompt_prefix(self) -> str:
        return "Extract key info from this:"
    
    def get_description(self) -> str:
        return "Extracts and highlights the most important information from text"

class AgentManager:
    """Manages multiple agents and handles routing"""
    
    def __init__(self, use_mock_llm: bool = False):
        # Initialize LLM client
        try:
            if use_mock_llm:
                raise ValueError("Using mock LLM")
            self.llm_client = LLMClient()
            # Test connection
            if not self.llm_client.test_connection():
                print("LLM API connection failed, switching to mock LLM")
                self.llm_client = MockLLM()
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            print("Using mock LLM for demonstration")
            self.llm_client = MockLLM()
        
        # Initialize memory cache
        self.memory_cache = MemoryCache()
        
        # Initialize agents
        self.agents = {
            'summarization': SummarizationAgent(self.llm_client, self.memory_cache),
            'planning': PlanningAgent(self.llm_client, self.memory_cache),
            'retrieval': RetrievalAgent(self.llm_client, self.memory_cache)
        }
    
    def get_available_agents(self) -> Dict[str, str]:
        """Get list of available agents with descriptions"""
        return {
            name: agent.get_description() 
            for name, agent in self.agents.items()
        }
    
    def process_query(self, agent_name: str, user_query: str) -> Dict[str, Any]:
        """Process a query using the specified agent"""
        if agent_name not in self.agents:
            return {
                'response': f"Unknown agent: {agent_name}",
                'is_cache_hit': False,
                'similarity_score': 0.0,
                'agent': 'error',
                'error': True
            }
        
        agent = self.agents[agent_name]
        result = agent.process_query(user_query)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.memory_cache.get_stats()
    
    def clear_cache(self):
        """Clear the memory cache"""
        self.memory_cache.clear_cache()
    
    def switch_llm_model(self, model_name: str) -> bool:
        """Switch the LLM model for all agents"""
        return self.llm_client.switch_model(model_name)
    
    def get_current_model(self) -> str:
        """Get the current LLM model name"""
        return self.llm_client.get_current_model()