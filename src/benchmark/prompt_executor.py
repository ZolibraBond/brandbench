"""Prompt executor for running benchmarks with optional web search."""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm


class PromptExecutor:
    """Execute prompts with or without web search functionality."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the prompt executor."""
        load_dotenv()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = self.config['openai']['model']
        self.temperature = self.config['openai']['temperature']
    
    def load_prompts(self, prompt_file: str) -> Dict[str, List[str]]:
        """Load prompts from a YAML file."""
        with open(prompt_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Skip comment lines and return the categorized prompts
        return {k: v for k, v in data.items() if not k.startswith('#')}
    
    async def execute_prompt(
        self,
        prompt: str,
        enable_web_search: bool = False,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a single prompt with optional web search.
        
        Args:
            prompt: The user prompt to execute
            enable_web_search: Whether to enable web search functionality
            system_message: Optional system message to guide the model
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Build the input for the Responses API
        if system_message:
            input_messages = [
                {"role": "developer", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            input_messages = [{"role": "user", "content": prompt}]
        
        # Prepare kwargs for the Responses API
        kwargs = {
            "model": self.model,
            "input": input_messages,
            "temperature": self.temperature
        }
        
        # Add web search tool if enabled
        if enable_web_search:
            kwargs["tools"] = [{"type": "web_search_preview"}]
        
        # Execute the prompt
        start_time = datetime.now()
        response = await self.client.responses.create(**kwargs)
        response_dict = response.model_dump()
        end_time = datetime.now()
        
        # Extract relevant information
        result = {
            "prompt": prompt,
            "web_search_enabled": enable_web_search,
            "timestamp": start_time.isoformat(),
            "response_time": (end_time - start_time).total_seconds(),
            "model": response_dict.get("model", self.model),
            "usage": response_dict.get("usage", {})
        }
        
        # Extract content from the response
        # The Responses API provides an output_text convenience property
        if hasattr(response, 'output_text'):
            result["content"] = response.output_text
        else:
            result["content"] = ""
        
        # Extract detailed output information
        output = response_dict.get("output", [])
        result["output"] = output
        
        # Extract web search information if available
        if enable_web_search and output:
            cited_urls = []
            for item in output:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            annotations = content.get("annotations", [])
                            for ann in annotations:
                                if ann.get("type") == "url_citation":
                                    cited_urls.append(ann["url_citation"]["url"])
            result["cited_urls"] = cited_urls
        else:
            result["cited_urls"] = []
        
        return result
    
    async def execute_batch(
        self,
        prompts: List[str],
        enable_web_search: bool = False,
        system_message: Optional[str] = None,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of prompts in parallel using async.
        
        Args:
            prompts: List of prompts to execute
            enable_web_search: Whether to enable web search
            system_message: Optional system message
            batch_size: Number of prompts to process concurrently
            
        Returns:
            List of results for each prompt
        """
        results = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(batch_size)
        
        async def execute_with_semaphore(prompt: str, index: int):
            async with semaphore:
                try:
                    result = await self.execute_prompt(prompt, enable_web_search, system_message)
                    return index, result
                except Exception as e:
                    return index, {
                        "prompt": prompt,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
        
        # Create all tasks
        tasks = [execute_with_semaphore(prompt, i) for i, prompt in enumerate(prompts)]
        
        # Execute with progress bar
        pbar = tqdm(total=len(prompts), desc="Processing prompts")
        
        # Process results as they complete
        results_dict = {}
        for coro in asyncio.as_completed(tasks):
            index, result = await coro
            results_dict[index] = result
            pbar.update(1)
        
        pbar.close()
        
        # Sort results by index to maintain order
        results = [results_dict[i] for i in range(len(prompts))]
        
        return results
    
    
    async def execute_prompt_file(
        self,
        prompt_file: str,
        enable_web_search: bool = False,
        system_message: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute all prompts from a file.
        
        Args:
            prompt_file: Path to the YAML prompt file
            enable_web_search: Whether to enable web search
            system_message: Optional system message
            categories: Optional list of categories to filter
            
        Returns:
            Dictionary mapping categories to their results
        """
        prompts_by_category = self.load_prompts(prompt_file)
        results = {}
        
        for category, prompts in prompts_by_category.items():
            if categories and category not in categories:
                continue
            
            print(f"\nProcessing category: {category}")
            results[category] = await self.execute_batch(
                prompts, enable_web_search, system_message
            )
        
        return results