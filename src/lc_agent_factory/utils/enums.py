# --------------------------------------------------------------------------------------------------------------------------------------
from enum import Enum
class SimpleEnum(Enum):
    def __get__(self, instance, owner):
        return self.value
    @classmethod
    def has(cls, value):
        return value in (e.value for e in cls)

# --------------------------------------------------------------------------------------------------------------------------------------

# LLM providers:
class LLMProviderSE(SimpleEnum):
    GOOGLE_GENAI = 'google_genai'
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    AZURE_OPENAI = 'azure_openai'
    AZURE_AI = 'azure_ai'
    GOOGLE_VERTEXAI = 'google_vertexai'
    ANTHROPIC_BEDROCK = 'anthropic_bedrock'
    BEDROCK = 'bedrock'
    BEDROCK_CONVERSE = 'bedrock_converse'
    COHERE = 'cohere'
    FIREWORKS = 'fireworks'
    TOGETHER = 'together'
    MISTRALAI = 'mistralai'
    HUGGINGFACE = 'huggingface'
    GROQ = 'groq'
    OLLAMA = 'ollama'
    GOOGLE_ANTHROPIC_VERTEX = 'google_anthropic_vertex'
    DEEPSEEK = 'deepseek'
    IBM = 'ibm'
    NVIDIA = 'nvidia'
    XAI = 'xai'
    OPENROUTER = 'openrouter'
    PERPLEXITY = 'perplexity'
    UPSTAGE = 'upstage'

# Complex Prebuilt Middleware
class ComplexPrebuiltMiddlewareSE(SimpleEnum):
    MODEL_FALLBACK = 'ModelFallbackMiddleware'
    CONTEXT_EDITING = 'ContextEditingMiddleware'
    SUMMARIZATION = 'SummarizationMiddleware'