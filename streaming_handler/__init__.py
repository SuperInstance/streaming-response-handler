"""
Streaming Response Handler - Handle streaming LLM responses elegantly.

Features:
- Unified interface for streaming from multiple providers
- Buffer management and chunk aggregation
- Callback and async event support
- Parse partial JSON responses
- Handle SSE (Server-Sent Events) streams
- Token counting during streaming
"""

from .handler import (
    StreamingHandler,
    StreamChunk,
    StreamCallback,
    StreamOptions,
    AsyncStreamHandler,
    stream_response,
    astream_response,
)
from .parsers import (
    JsonStreamParser,
    parse_partial_json,
)
from .buffer import (
    StreamBuffer,
    ChunkBuffer,
)

__version__ = "1.0.0"
__all__ = [
    "StreamingHandler",
    "StreamChunk",
    "StreamCallback",
    "StreamOptions",
    "AsyncStreamHandler",
    "stream_response",
    "astream_response",
    "JsonStreamParser",
    "parse_partial_json",
    "StreamBuffer",
    "ChunkBuffer",
]
