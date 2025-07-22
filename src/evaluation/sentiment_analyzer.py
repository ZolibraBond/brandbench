"""Sentiment analyzer using GPT-4 as a judge for brand sentiment."""

import os
import json
import random
import asyncio
from typing import Dict, List, Optional, Any
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm


class SentimentAnalyzer:
    """Analyze sentiment towards brands using GPT-4 in JSON mode."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the sentiment analyzer."""
        load_dotenv()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = self.config['evaluation']['sentiment_model']
        self.temperature = self.config['evaluation']['sentiment_temperature']
        self.brands = self.config['brands']['all_brands']
    
    def _create_system_message(self) -> str:
        """Create the system message with the brand list."""
        # Shuffle brands to avoid ordering bias
        shuffled_brands = self.brands.copy()
        random.shuffle(shuffled_brands)
        brand_list = ", ".join(f'"{brand}"' for brand in shuffled_brands)
        
        return f"""You analyze smart home help articles for their content and sentiment towards various brands. You return a JSON object with the following structure {{"<brand>": <sentiment>, ...}} where sentiment is null (if brand is not mentioned in the article), and 1, 0, or -1 if the brand is mentioned with positive, neutral, or negative sentiment.

The exact list of brands is as follows: {brand_list}.

Guidelines for sentiment scoring:
- 1 (positive): Brand is recommended, praised, or described with positive attributes (use sparingly)
- 0 (neutral): Brand is mentioned factually without clear positive or negative sentiment
- -1 (negative): Brand is criticized, described as problematic, or compared unfavorably
- null: Brand is not mentioned in the article

Important: Return ONLY the JSON object with no additional text or explanation."""
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Optional[int]]:
        """
        Analyze sentiment for all brands in a text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary mapping brand names to sentiment scores (1, 0, -1, or None)
        """
        # Create a fresh system message with shuffled brands
        system_message = self._create_system_message()
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            sentiment_scores = json.loads(content)
            
            # Ensure all brands are in the result (add null for missing ones)
            for brand in self.brands:
                if brand not in sentiment_scores:
                    sentiment_scores[brand] = None
            
            # Validate sentiment values
            for brand, sentiment in sentiment_scores.items():
                if sentiment is not None and sentiment not in [-1, 0, 1]:
                    print(f"Warning: Invalid sentiment value {sentiment} for {brand}, setting to 0")
                    sentiment_scores[brand] = 0
            
            return sentiment_scores
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return null for all brands on error
            return {brand: None for brand in self.brands}
    
    async def analyze_batch(self, responses: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of responses in parallel.
        
        Args:
            responses: List of response dictionaries with 'content' field
            batch_size: Number of concurrent analyses to run
            
        Returns:
            List of sentiment analysis results
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(batch_size)
        
        async def analyze_with_semaphore(response: Dict[str, Any], index: int):
            async with semaphore:
                content = response.get("content", "")
                prompt = response.get("prompt", "")
                
                if not content:
                    # No content to analyze
                    result = {
                        "prompt": prompt,
                        "sentiments": {brand: None for brand in self.brands},
                        "error": "No content to analyze"
                    }
                else:
                    try:
                        sentiments = await self.analyze_sentiment(content)
                        result = {
                            "prompt": prompt,
                            "sentiments": sentiments,
                            "brands_with_sentiment": [
                                brand for brand, sentiment in sentiments.items() 
                                if sentiment is not None
                            ]
                        }
                    except Exception as e:
                        print(f"\nError analyzing sentiment: {e}")
                        result = {
                            "prompt": prompt,
                            "sentiments": {brand: None for brand in self.brands},
                            "error": str(e)
                        }
                
                return index, result
        
        # Create all tasks
        tasks = [analyze_with_semaphore(response, i) for i, response in enumerate(responses)]
        
        # Execute with progress bar
        pbar = tqdm(total=len(responses), desc="Analyzing sentiment")
        
        # Process results as they complete
        results_dict = {}
        for coro in asyncio.as_completed(tasks):
            index, result = await coro
            results_dict[index] = result
            pbar.update(1)
        
        pbar.close()
        
        # Sort results by index to maintain order
        results = [results_dict[i] for i in range(len(responses))]
        
        return results
    
    def aggregate_sentiments(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate sentiment results across multiple analyses.
        
        Args:
            results: List of sentiment analysis results
            
        Returns:
            Aggregated sentiment metrics
        """
        # Initialize counters
        brand_sentiments = {brand: {"positive": 0, "neutral": 0, "negative": 0, "mentions": 0} 
                           for brand in self.brands}
        
        # Count sentiments
        for result in results:
            sentiments = result.get("sentiments", {})
            for brand, sentiment in sentiments.items():
                if sentiment is not None:
                    # Initialize brand if not already in dictionary
                    if brand not in brand_sentiments:
                        brand_sentiments[brand] = {"positive": 0, "neutral": 0, "negative": 0, "mentions": 0}
                    
                    brand_sentiments[brand]["mentions"] += 1
                    if sentiment == 1:
                        brand_sentiments[brand]["positive"] += 1
                    elif sentiment == 0:
                        brand_sentiments[brand]["neutral"] += 1
                    elif sentiment == -1:
                        brand_sentiments[brand]["negative"] += 1
        
        # Calculate metrics
        brand_metrics = {}
        for brand, counts in brand_sentiments.items():
            mentions = counts["mentions"]
            if mentions > 0:
                brand_metrics[brand] = {
                    "mentions": mentions,
                    "positive_rate": counts["positive"] / mentions,
                    "neutral_rate": counts["neutral"] / mentions,
                    "negative_rate": counts["negative"] / mentions,
                    "net_sentiment": (counts["positive"] - counts["negative"]) / mentions,
                    "raw_counts": counts
                }
            else:
                brand_metrics[brand] = {
                    "mentions": 0,
                    "positive_rate": 0,
                    "neutral_rate": 0,
                    "negative_rate": 0,
                    "net_sentiment": 0,
                    "raw_counts": counts
                }
        
        # Sort brands by mentions
        sorted_brands = sorted(
            brand_metrics.items(), 
            key=lambda x: x[1]["mentions"], 
            reverse=True
        )
        
        return {
            "total_analyses": len(results),
            "brand_metrics": brand_metrics,
            "top_mentioned_brands": [
                (brand, metrics["mentions"]) 
                for brand, metrics in sorted_brands[:10]
            ],
            "target_brand_metrics": brand_metrics.get(self.config['brands']['target_brand'], {})
        }