from langchain_openai import ChatOpenAI

from src.utils.llm.models.data import ModelResponse

# Only OpenAI is imported up front. The other providers (Vertex/Claude/Gemini,
# Together, DeepSeek, vLLM) are imported lazily inside get_engine so a deploy
# that only uses OpenAI doesn't have to ship those heavy dependencies.

# OpenAI chat models we use directly.
openai_models = {
    "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini-2024-07-18",
    "gpt-3.5-turbo-0125", "gpt-4o",
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2",
}

# The GPT-5 models don't take max_tokens or a custom temperature like the older
# OpenAI models do, so they need to be handled a little differently below.
gpt5_models = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2"}


def get_engine(model_name, **kwargs):
    """
    Creates and returns a language model engine based on the specified model name.

    Args:
        model_name (str): Name of the model to initialize. Supported models:
            - OpenAI models: gpt-4o-mini, gpt-4o, gpt-4.1-mini, gpt-5, gpt-5-mini, ...
            - Llama models (Together): meta-llama/Llama-3.1-8B-Instruct, ...
            - DeepSeek models: deepseek-ai/DeepSeek-V3
            - Claude / Gemini models: via Vertex AI
            - vLLM models: prefix with "vllm:" (requires VLLM_BASE_URL)
        **kwargs: Additional keyword arguments to pass to the model constructor.
            - temperature: Float between 0 and 1 (default: 0.0)
            - max_tokens/max_output_tokens: token limit (default: 8192), mapped to
              the right parameter name for each provider.

    Returns:
        A chat model / custom engine. All engines return ModelResponse objects.
    """
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0

    max_tokens = kwargs.pop("max_tokens", None)
    max_output_tokens = kwargs.pop("max_output_tokens", None)
    max_tokens_to_sample = kwargs.pop("max_tokens_to_sample", None)
    token_limit = max_output_tokens or max_tokens or max_tokens_to_sample or 8192

    if model_name == "gpt-4o-mini":
        model_name = "gpt-4o-mini-2024-07-18"

    # Claude via Vertex AI
    if "claude" in model_name:
        from src.utils.llm.models.claude import ClaudeVertexEngine
        kwargs["max_tokens_to_sample"] = token_limit
        return ClaudeVertexEngine(model_name=model_name, **kwargs)

    # Gemini via Vertex AI
    if "gemini" in model_name:
        from src.utils.llm.models.gemini import GeminiVertexEngine
        kwargs["max_output_tokens"] = token_limit
        return GeminiVertexEngine(model_name=model_name, **kwargs)

    # DeepSeek
    if "deepseek" in model_name.lower():
        from src.utils.llm.models.deepseek import DeepSeekEngine
        kwargs["max_tokens"] = token_limit
        return DeepSeekEngine(model_name=model_name, **kwargs)

    # vLLM (identified by the vllm: prefix)
    if model_name.startswith("vllm:"):
        from src.utils.llm.models.vllm import VLLMEngine
        kwargs["max_tokens"] = token_limit
        return VLLMEngine(model_name=model_name[5:], **kwargs)

    # GPT-5 models: use max_completion_tokens and only the default temperature (1).
    if model_name in gpt5_models or model_name.startswith("gpt-5"):
        kwargs["temperature"] = 1
        model_kwargs = kwargs.pop("model_kwargs", {})
        model_kwargs["max_completion_tokens"] = token_limit
        kwargs["model_kwargs"] = model_kwargs
        kwargs["model_name"] = model_name
        return ChatOpenAI(**kwargs)

    # Everything else uses max_tokens.
    kwargs["max_tokens"] = token_limit
    kwargs["model_name"] = model_name

    if model_name in openai_models:
        return ChatOpenAI(**kwargs)

    # Fall back to Together (Llama and friends).
    from langchain_together import ChatTogether
    return ChatTogether(**kwargs)


def invoke_engine(engine, prompt, **kwargs) -> ModelResponse:
    """Invoke an engine and return a ModelResponse (with token usage if available)."""
    response = engine.invoke(prompt, **kwargs)

    # Custom engines already return a ModelResponse.
    if isinstance(response, ModelResponse):
        return response

    model_response = ModelResponse(response.content)
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        model_response.response_metadata = {
            'token_usage': response.response_metadata['token_usage']
        }
    return model_response
