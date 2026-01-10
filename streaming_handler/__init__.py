"""
Streaming Response Handler - Handle streaming LLM responses elegantly.

Features:
- Unified interface for streaming from multiple providers
- Buffer management and chunk aggregation
- Callback and async event support
- Parse partial JSON responses
- Handle SSE (Server-Sent Events) streams
- Token counting during streaming
- Automatic retry with exponential backoff
"""

from .handler import (
    StreamingHandler,
    StreamChunk,
    StreamCallback,
    StreamOptions,
    StreamEventType,
    AsyncStreamHandler,
    stream_response,
    astream_response,
)
from .parsers import (
    JsonStreamParser,
    parse_partial_json,
    extract_sse_data,
    parse_sse_chunk,
    SSEParser,
)
from .buffer import (
    StreamBuffer,
    ChunkBuffer,
    TokenBuffer,
    BufferedChunk,
    BufferFullError,
)
from .retry import (
    RetryConfig,
    RetryStrategy,
    RetryResult,
    RetryableStreamIterator,
    AsyncRetryableStreamIterator,
    retry_stream,
    aretry_stream,
    retry_decorator,
    DEFAULT_RETRY,
    AGGRESSIVE_RETRY,
    CONSERVATIVE_RETRY,
    RATE_LIMIT_RETRY,
)

__version__ = "1.1.0"
__all__ = [
    "StreamingHandler",
    "StreamChunk",
    "StreamCallback",
    "StreamOptions",
    "StreamEventType",
    "AsyncStreamHandler",
    "stream_response",
    "astream_response",
    "JsonStreamParser",
    "parse_partial_json",
    "extract_sse_data",
    "parse_sse_chunk",
    "SSEParser",
    "StreamBuffer",
    "ChunkBuffer",
    "TokenBuffer",
    "BufferedChunk",
    "BufferFullError",
    # Retry exports
    "RetryConfig",
    "RetryStrategy",
    "RetryResult",
    "RetryableStreamIterator",
    "AsyncRetryableStreamIterator",
    "retry_stream",
    "aretry_stream",
    "retry_decorator",
    "DEFAULT_RETRY",
    "AGGRESSIVE_RETRY",
    "CONSERVATIVE_RETRY",
    "RATE_LIMIT_RETRY",
]
