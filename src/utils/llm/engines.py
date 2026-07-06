
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import VertexAI

from src.utils.llm.models.data import ModelResponse
from src.utils.llm.models.claude import ClaudeVertexEngine, claude_vertex_model_mapping
from src.utils.llm.models.gemini import GeminiVertexEngine, gemini_models
from src.utils.llm.models.deepseek import DeepSeekEngine, deepseek_models
from src.utils.llm.models.vllm import VLLMEngine



engine_constructor = {
    "gpt-4.1-mini": ChatOpenAI,
    "gpt-4.1-nano": ChatOpenAI,
    "gpt-4o-mini-2024-07-18": ChatOpenAI,
    "gpt-3.5-turbo-0125": ChatOpenAI,
    "gpt-4o": ChatOpenAI,
    "gpt-5": ChatOpenAI,
    "gpt-5-mini": ChatOpenAI,
    "gpt-5-nano": ChatOpenAI,
    "gpt-5.1": ChatOpenAI,
    "gpt-5.2": ChatOpenAI,
    "meta-llama/Llama-3.1-8B-Instruct": ChatTogether,
    "meta-llama/Llama-3.1-70B-Instruct": ChatTogether
}

# The GPT-5 models don't take max_tokens or a custom temperature like the older
# OpenAI models do, so they need to be handled a little differently below.
gpt5_models = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2"}

def get_engine(model_name, **kwargs):
    """
    Creates and returns a language model engine based on the specified model name.

    Args:
        model_name (str): Name of the model to initialize. Supported models:
            - OpenAI models: gpt-4o-mini, gpt-3.5-turbo-0125, gpt-4o
            - Llama models: meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.1-70B-Instruct
            - DeepSeek models: deepseek-ai/DeepSeek-V3 (671B parameter model)
            - Claude models: via Vertex AI
            - Gemini models: via Vertex AI
            - vLLM models: prefix with "vllm:" (e.g., vllm:meta-llama/Llama-3.1-8B-Instruct)
                Requires VLLM_BASE_URL environment variable
        **kwargs: Additional keyword arguments to pass to the model constructor.
            - temperature: Float between 0 and 1 (default: 0.0)
            - max_tokens/max_output_tokens: Maximum number of tokens in the response (default: 4096)
                Note: This will be mapped to the appropriate parameter name for each model:
                - OpenAI/Llama/DeepSeek/vLLM: max_tokens
                - Gemini: max_output_tokens
                - Claude: max_tokens_to_sample

    Returns:
        LangChain chat model instance or custom engine configured with the specified parameters
        All engines now return ModelResponse objects with token usage metadata populated
    """
    # Set default temperature if not provided
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0

    # Standardize max token handling
    max_tokens = kwargs.pop("max_tokens", None)
    max_output_tokens = kwargs.pop("max_output_tokens", None)
    max_tokens_to_sample = kwargs.pop("max_tokens_to_sample", None)
    
    # Use the first non-None value in order of precedence
    token_limit = max_output_tokens or max_tokens or max_tokens_to_sample or 8192
        
    if model_name == "gpt-4o-mini":
        model_name = "gpt-4o-mini-2024-07-18"
    
    # Handle Claude models via Vertex AI
    if model_name in claude_vertex_model_mapping or "claude" in model_name:
        kwargs["max_tokens_to_sample"] = token_limit
        return ClaudeVertexEngine(model_name=model_name, **kwargs)
    
    # Handle Gemini models via Vertex AI
    if model_name in gemini_models or "gemini" in model_name:
        kwargs["max_output_tokens"] = token_limit
        return GeminiVertexEngine(model_name=model_name, **kwargs)
        
    # Handle DeepSeek models
    if model_name in deepseek_models or "deepseek" in model_name.lower():
        kwargs["max_tokens"] = token_limit
        return DeepSeekEngine(model_name=model_name, **kwargs)

    # Handle vLLM models (identified by vllm: prefix)
    if model_name.startswith("vllm:"):
        # Extract the actual model name after the vllm: prefix
        actual_model_name = model_name[5:]  # Remove "vllm:" prefix
        kwargs["max_tokens"] = token_limit
        return VLLMEngine(model_name=actual_model_name, **kwargs)

    # GPT-5 models: they use max_completion_tokens instead of max_tokens and
    # only allow the default temperature (1), so set those and skip the usual
    # max_tokens below.
    if model_name in gpt5_models or model_name.startswith("gpt-5"):
        kwargs["temperature"] = 1
        model_kwargs = kwargs.pop("model_kwargs", {})
        model_kwargs["max_completion_tokens"] = token_limit
        kwargs["model_kwargs"] = model_kwargs
        kwargs["model_name"] = model_name
        return ChatOpenAI(**kwargs)

    # For other models (OpenAI, Llama), use max_tokens
    kwargs["max_tokens"] = token_limit
    kwargs["model_name"] = model_name
    return engine_constructor[model_name](**kwargs)

def invoke_engine(engine, prompt, **kwargs) -> ModelResponse:
    """
    Simple wrapper to invoke a language model engine and return its response.

    Args:
        engine: The language model engine to use
        prompt: The input prompt to send to the model
        **kwargs: Additional keyword arguments for the model invocation

    Returns:
        ModelResponse: The model's response with content and token usage metadata
    """
    response = engine.invoke(prompt, **kwargs)

    # If the engine is a custom engine (returns ModelResponse already), return it directly
    if isinstance(response, ModelResponse):
        return response

    # For LangChain models, wrap the response and extract token usage
    model_response = ModelResponse(response.content)

    # Extract token usage from LangChain response metadata if available
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        model_response.response_metadata = {
            'token_usage': response.response_metadata['token_usage']
        }

    return model_response
