"""
Token cost utilities.

Provides a model price table and cost calculation for cost tracking during training.
"""

from typing import Dict, Optional
import logging

# Use the standard-library logging to avoid circular imports
logger = logging.getLogger(__name__)

# Price table (USD per 1K tokens), based on OpenAI and other major LLM providers
TOKEN_COSTS: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {
        "prompt":  0.005,       # $ per 1K tokens
        "completion": 0.015,    # $ per 1K tokens
    },
    "gpt-4o-mini": {
        "prompt": 0.00015,     # $0.00015 per 1K tokens
        "completion": 0.0006   # $0.0006 per 1K tokens
    },
    "gpt-4": {
        "prompt": 0.03,        # $0.03 per 1K tokens
        "completion": 0.06     # $0.06 per 1K tokens
    },
    "gpt-4-turbo": {
        "prompt": 0.01,        # $0.01 per 1K tokens
        "completion": 0.03     # $0.03 per 1K tokens
    },
    "gpt-3.5-turbo": {
        "prompt": 0.0005,      # $0.0005 per 1K tokens
        "completion": 0.0015   # $0.0015 per 1K tokens
    },
    "gpt-4.1": {  # custom gpt-4.1 uses gpt-4o pricing
        "prompt": 0.005,
        "completion": 0.015
    },
}


class CostManager:
    """Cost manager for computing and tracking model call costs."""
    
    def __init__(self, token_costs: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the cost manager.
        
        Args:
            token_costs: Custom price table (defaults to TOKEN_COSTS if None)
        """
        self.token_costs = token_costs if token_costs is not None else TOKEN_COSTS
    
    def get_model_prices(self, model_id: str) -> Optional[Dict[str, float]]:
        """
        Get the prices for a model.
        
        Args:
            model_id: Model ID (e.g. "gpt-4o-mini")
            
        Returns:
            Price dict {"prompt": float, "completion": float}, or None if the model is unknown
        """
        return self.token_costs.get(model_id)
    
    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Compute the cost of a model call.
        
        Formula: cost = (input_tokens * prompt_price + output_tokens * completion_price) / 1000
        
        Args:
            model_id: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD; falls back to (input_tokens + output_tokens) / 1000 if the model is not in the price table
        """
        prices = self.get_model_prices(model_id)
        
        if prices is None:
            logger.warning(
                f"Model '{model_id}' not found in price table. "
                f"Using default cost calculation (token count / 1000)."
            )
            # Model not in price table: fall back to token count as cost
            return float(input_tokens + output_tokens) / 1000.0
        
        prompt_price = prices["prompt"]
        completion_price = prices["completion"]
        
        # cost = (input_tokens * prompt_price + output_tokens * completion_price) / 1000
        cost = (input_tokens * prompt_price + output_tokens * completion_price) / 1000.0
        
        return cost


# Global cost manager instance (default price table)
_default_cost_manager = CostManager()


def get_cost_manager() -> CostManager:
    """
    Get the global cost manager instance.
    
    Returns:
        CostManager instance
    """
    return _default_cost_manager


def calculate_agent_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cost_manager: Optional[CostManager] = None
) -> float:
    """
    Compute the cost of an agent run (convenience wrapper).
    
    Args:
        model_id: Model ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_manager: Cost manager (uses the global instance if None)
        
    Returns:
        Cost in USD
    """
    if cost_manager is None:
        cost_manager = _default_cost_manager
    
    return cost_manager.calculate_cost(model_id, input_tokens, output_tokens)

