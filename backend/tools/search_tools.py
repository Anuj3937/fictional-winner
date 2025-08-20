import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import json
import re
from dataclasses import dataclass
from cachetools import TTLCache
from loguru import logger
import os

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0

class StatisticalSearchEngine:
    """High-performance statistical data search with caching"""
    
    def __init__(self):
        # Cache search results for 1 hour to improve performance
        self.cache: TTLCache[str, List[SearchResult]] = TTLCache(
            maxsize=1000, 
            ttl=3600
        )
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def search_statistical_data(self, domain: str, task_type: str) -> Dict[str, Any]:
        """Search for statistical characteristics with O(1) cache lookup"""
        
        cache_key = f"{domain}_{task_type}_stats"
        
        # Check cache first - O(1)
        if cache_key in self.cache:
            logger.info(f"Using cached results for {cache_key}")
            cached_results = self.cache[cache_key]
            return self._process_search_results(cached_results, domain, task_type)
        
        # Perform search if not cached
        search_queries = [
            f"{domain} dataset typical features statistics mean std",
            f"{domain} {task_type} benchmark dataset characteristics",
            f"real world {domain} data distribution parameters"
        ]
        
        all_results = []
        session = await self._get_session()
        
        # Perform searches concurrently for speed
        tasks = [
            self._perform_search(session, query) 
            for query in search_queries
        ]
        
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for results in search_results:
            if isinstance(results, list):
                all_results.extend(results)
        
        # Cache results
        self.cache[cache_key] = all_results
        
        return self._process_search_results(all_results, domain, task_type)
    
    async def _perform_search(self, session: aiohttp.ClientSession, query: str) -> List[SearchResult]:
        """Perform actual search with error handling"""
        
        try:
            if os.getenv("SERPAPI_KEY"):
                return await self._serpapi_search(session, query)
            elif os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
                return await self._google_search(session, query)
            else:
                logger.warning("No search API keys configured, using mock data")
                return self._get_mock_search_results(query)
                
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def _serpapi_search(self, session: aiohttp.ClientSession, query: str) -> List[SearchResult]:
        """Search using SerpAPI"""
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "engine": "google",
            "num": 6
        }
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("organic_results", [])[:6]:
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        relevance_score=self._calculate_relevance(item.get("snippet", ""), query)
                    )
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []
    
    async def _google_search(self, session: aiohttp.ClientSession, query: str) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": os.getenv("GOOGLE_API_KEY"),
            "cx": os.getenv("GOOGLE_CSE_ID"),
            "q": query,
            "num": 6
        }
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("items", []):
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        relevance_score=self._calculate_relevance(item.get("snippet", ""), query)
                    )
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Google Search failed: {e}")
            return []
    
    def _get_mock_search_results(self, query: str) -> List[SearchResult]:
        """Generate mock search results for development"""
        
        mock_results = [
            SearchResult(
                title=f"Statistical Analysis of {query}",
                url="https://example.com/stats",
                snippet=f"Typical {query} datasets have mean values around 50-100, standard deviation 10-20, with normal distribution patterns commonly observed.",
                relevance_score=0.8
            ),
            SearchResult(
                title=f"Benchmark Study: {query}",
                url="https://example.com/benchmark", 
                snippet=f"Research shows {query} data typically ranges from 0-1000, with 75% of values between 20-200, median around 80.",
                relevance_score=0.7
            )
        ]
        
        return mock_results
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score efficiently"""
        if not text or not query:
            return 0.0
        
        text_lower = text.lower()
        query_words = query.lower().split()
        
        # Count matches - O(n) where n is text length
        matches = sum(1 for word in query_words if word in text_lower)
        
        # Bonus for statistical terms
        statistical_terms = ["mean", "average", "std", "standard deviation", "distribution", "range", "median"]
        stat_matches = sum(1 for term in statistical_terms if term in text_lower)
        
        # Bonus for numerical values
        numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
        
        # Calculate score
        base_score = matches / len(query_words) if query_words else 0
        bonus = (stat_matches * 0.1) + (min(numbers, 5) * 0.05)
        
        return min(1.0, base_score + bonus)
    
    def _process_search_results(self, results: List[SearchResult], domain: str, task_type: str) -> Dict[str, Any]:
        """Process search results to extract statistical parameters"""
        
        if not results:
            return self._get_default_stats(domain, task_type)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Extract statistical information from top results
        statistical_params = {
            "feature_ranges": {},
            "distributions": {},
            "typical_values": {},
            "correlations": []
        }
        
        for result in results[:5]:  # Use top 5 results
            text = result.snippet
            
            # Extract numerical ranges using regex - O(n) where n is text length
            ranges = re.findall(r'(\w+)[^\d]*(\d+(?:\.\d+)?)[^\d]*(?:to|-|and)[^\d]*(\d+(?:\.\d+)?)', text.lower())
            for feature, min_val, max_val in ranges:
                try:
                    statistical_params["feature_ranges"][feature] = {
                        "min": float(min_val),
                        "max": float(max_val),
                        "source": result.url
                    }
                except ValueError:
                    continue
            
            # Extract mean values
            means = re.findall(r'(?:mean|average)[^\d]*(\d+(?:\.\d+)?)', text.lower())
            if means:
                statistical_params["typical_values"]["mean"] = float(means[0])
            
            # Extract standard deviations
            stds = re.findall(r'(?:std|standard deviation)[^\d]*(\d+(?:\.\d+)?)', text.lower())
            if stds:
                statistical_params["typical_values"]["std"] = float(stds)
            
            # Detect distribution types
            if "normal" in text.lower() or "gaussian" in text.lower():
                statistical_params["distributions"]["primary"] = "normal"
            elif "uniform" in text.lower():
                statistical_params["distributions"]["primary"] = "uniform"
            elif "exponential" in text.lower():
                statistical_params["distributions"]["primary"] = "exponential"
        
        # Add domain-specific defaults if nothing found
        if not statistical_params["feature_ranges"]:
            statistical_params.update(self._get_default_stats(domain, task_type))
        
        return statistical_params
    
    def _get_default_stats(self, domain: str, task_type: str) -> Dict[str, Any]:
        """Get default statistical parameters for domain"""
        
        domain_defaults = {
            "healthcare": {
                "feature_ranges": {
                    "age": {"min": 18, "max": 85},
                    "weight": {"min": 40, "max": 150},
                    "height": {"min": 150, "max": 200},
                    "blood_pressure": {"min": 90, "max": 180},
                    "heart_rate": {"min": 50, "max": 120}
                },
                "distributions": {"primary": "normal"},
                "typical_values": {"mean": 50, "std": 15}
            },
            "finance": {
                "feature_ranges": {
                    "income": {"min": 20000, "max": 200000},
                    "credit_score": {"min": 300, "max": 850},
                    "age": {"min": 18, "max": 70},
                    "debt_ratio": {"min": 0, "max": 1}
                },
                "distributions": {"primary": "log-normal"},
                "typical_values": {"mean": 65000, "std": 25000}
            },
            "retail": {
                "feature_ranges": {
                    "price": {"min": 1, "max": 1000},
                    "quantity": {"min": 1, "max": 100},
                    "rating": {"min": 1, "max": 5},
                    "discount": {"min": 0, "max": 0.5}
                },
                "distributions": {"primary": "normal"},
                "typical_values": {"mean": 50, "std": 20}
            }
        }
        
        return domain_defaults.get(domain, domain_defaults["healthcare"])
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

# Global search engine instance
search_engine = StatisticalSearchEngine()
