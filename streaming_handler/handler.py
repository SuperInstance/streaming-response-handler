"""
Core streaming response handler for LLM APIs.
"""

import asyncio
import json
from typing import (
    Dict,
    List,
    Optional,
    Callable,
    AsyncIterator,
    Iterator,
    Any,
    Union,
)
from dataclasses import dataclass, field
from enum import Enum
import time


class StreamEventType(str, Enum):
    """Types of events in a stream"""
    TOKEN = "token"
    CHUNK = "chunk"
    METADATA = "metadata"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamChunk:
    """A chunk of data from a streaming response"""
    content: str
    delta: str  # Just the new content since last chunk
    is_first: bool = False
    is_last: bool = False
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Estimate token count (rough approximation)"""
        return len(self.content) // 4


@dataclass
class StreamOptions:
    """Options for streaming behavior"""
    buffer_size: int = 100  # Chunks to buffer before yielding
    include_metadata: bool = True
    yield_on_first_chunk: bool = True
    aggregate_tokens: bool = False
    token_aggregation_count: int = 10  # Aggregate N tokens before yielding


StreamCallback = Callable[[StreamChunk], None]
AsyncStreamCallback = Callable[[StreamChunk], Any]


class StreamingHandler:
    """
    Handle streaming LLM responses with buffering and callbacks.

    Example:
        def on_chunk(chunk: StreamChunk):
            print(chunk.delta, end='', flush=True)

        handler = StreamingHandler(callback=on_chunk)
        handler.process_stream(response_iterator)

        print(f"\\nFull response: {handler.get_full_content()}")
        print(f"Total chunks: {handler.chunk_count}")
        print(f"Estimated tokens: {handler.estimated_tokens}")
    """

    def __init__(
        self,
        callback: Optional[StreamCallback] = None,
        options: Optional[StreamOptions] = None
    ):
        """
        Initialize the streaming handler.

        Args:
            callback: Optional callback for each chunk
            options: Stream options
        """
        self.callback = callback
        self.options = options or StreamOptions()

        # State
        self.chunks: List[StreamChunk] = []
        self._content_buffer: List[str] = []
        self._token_buffer: List[str] = []
        self._is_first = True
        self._is_done = False

    def process_stream(
        self,
        stream: Iterator[str],
        metadata_parser: Optional[Callable[[str], Dict[str, Any]]] = None
    ) -> str:
        """
        Process a streaming response.

        Args:
            stream: Iterator yielding response chunks
            metadata_parser: Optional function to parse metadata from chunks

        Returns:
            Full accumulated content
        """
        for raw_chunk in stream:
            self._process_chunk(raw_chunk, metadata_parser)

        return self.get_full_content()

    def process_chunks(self, chunks: List[str]) -> str:
        """
        Process a list of chunks.

        Args:
            chunks: List of response chunks

        Returns:
            Full accumulated content
        """
        for chunk in chunks:
            self._process_chunk(chunk)

        return self.get_full_content()

    def add_chunk(self, content: str, **metadata) -> None:
        """
        Manually add a chunk.

        Args:
            content: Chunk content
            **metadata: Optional metadata (finish_reason, etc.)
        """
        finish_reason = metadata.get("finish_reason")
        is_last = finish_reason is not None

        chunk = StreamChunk(
            content=content,
            delta=content,
            is_first=self._is_first,
            is_last=is_last,
            finish_reason=finish_reason,
            metadata=metadata
        )

        self._add_chunk(chunk)

    def _process_chunk(
        self,
        raw_chunk: str,
        metadata_parser: Optional[Callable[[str], Dict[str, Any]]] = None
    ) -> None:
        """Process a single raw chunk."""
        # Parse metadata if provided
        metadata = {}
        if metadata_parser:
            try:
                metadata = metadata_parser(raw_chunk)
            except:
                pass

        # Extract content (this varies by provider format)
        # Default: treat entire chunk as content
        content = raw_chunk

        # Handle common SSE formats
        if "data: " in raw_chunk:
            for line in raw_chunk.split("\n"):
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        metadata["finish_reason"] = "done"
                    else:
                        try:
                            parsed = json.loads(data)
                            # OpenAI format
                            if "choices" in parsed:
                                choices = parsed["choices"]
                                if choices and "delta" in choices[0]:
                                    delta = choices[0]["delta"]
                                    content = delta.get("content", "")
                                    if "finish_reason" in choices[0]:
                                        metadata["finish_reason"] = choices[0]["finish_reason"]
                            # Anthropic format
                            elif "delta" in parsed:
                                content = parsed["delta"].get("text", "")
                                if parsed.get("type") == "message_stop":
                                    metadata["finish_reason"] = "end_turn"
                        except:
                            pass

        is_last = "finish_reason" in metadata
        chunk = StreamChunk(
            content=self._accumulated_content() + content,
            delta=content,
            is_first=self._is_first and content,
            is_last=is_last,
            finish_reason=metadata.get("finish_reason"),
            metadata=metadata
        )

        if content:
            self._is_first = False

        if is_last:
            self._is_done = True

        self._add_chunk(chunk)

    def _add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to state and trigger callback."""
        self.chunks.append(chunk)

        if chunk.delta:
            self._content_buffer.append(chunk.delta)

        # Trigger callback
        if self.callback:
            self.callback(chunk)

    def get_full_content(self) -> str:
        """Get the full accumulated content."""
        return "".join(self._content_buffer)

    def _accumulated_content(self) -> str:
        """Get current accumulated content."""
        return "".join(self._content_buffer)

    @property
    def chunk_count(self) -> int:
        """Get number of chunks received."""
        return len(self.chunks)

    @property
    def estimated_tokens(self) -> int:
        """Get estimated token count."""
        return len(self.get_full_content()) // 4

    @property
    def is_done(self) -> bool:
        """Check if stream is complete."""
        return self._is_done

    def reset(self) -> None:
        """Reset handler state for reuse."""
        self.chunks.clear()
        self._content_buffer.clear()
        self._is_first = True
        self._is_done = False

    def get_chunks_by_type(self, event_type: StreamEventType) -> List[StreamChunk]:
        """Get chunks filtered by type."""
        if event_type == StreamEventType.TOKEN:
            return [c for c in self.chunks if c.delta]
        elif event_type == StreamEventType.DONE:
            return [c for c in self.chunks if c.is_last]
        elif event_type == StreamEventType.METADATA:
            return [c for c in self.chunks if c.metadata]
        return []

    def get_timings(self) -> Dict[str, float]:
        """Get timing information if chunks have timestamps."""
        if not self.chunks:
            return {}

        # Check if chunks have timing metadata
        timed_chunks = [c for c in self.chunks if "timestamp" in c.metadata]
        if not timed_chunks:
            return {"chunk_count": len(self.chunks)}

        first = timed_chunks[0].metadata["timestamp"]
        last = timed_chunks[-1].metadata["timestamp"]
        return {
            "start_time": first,
            "end_time": last,
            "duration_ms": (last - first) * 1000,
            "chunks_per_second": len(self.chunks) / (last - first) if last > first else 0
        }


class AsyncStreamHandler:
    """
    Async version of StreamingHandler.

    Example:
        async def astream_response():
            async for chunk in fetch_stream():
                yield chunk

        handler = AsyncStreamHandler()
        await handler.process_stream(astream_response())
    """

    def __init__(
        self,
        callback: Optional[AsyncStreamCallback] = None,
        options: Optional[StreamOptions] = None
    ):
        """
        Initialize the async streaming handler.

        Args:
            callback: Optional async callback for each chunk
            options: Stream options
        """
        self.callback = callback
        self.options = options or StreamOptions()

        # State
        self.chunks: List[StreamChunk] = []
        self._content_buffer: List[str] = []
        self._is_first = True
        self._is_done = False

    async def process_stream(
        self,
        stream: AsyncIterator[str],
        metadata_parser: Optional[Callable[[str], Any]] = None
    ) -> str:
        """
        Process an async streaming response.

        Args:
            stream: Async iterator yielding response chunks
            metadata_parser: Optional function to parse metadata

        Returns:
            Full accumulated content
        """
        async for raw_chunk in stream:
            await self._process_chunk_async(raw_chunk, metadata_parser)

        return self.get_full_content()

    async def _process_chunk_async(
        self,
        raw_chunk: str,
        metadata_parser: Optional[Callable[[str], Any]] = None
    ) -> None:
        """Process a single raw chunk asynchronously."""
        # Parse metadata
        metadata = {}
        if metadata_parser:
            try:
                result = metadata_parser(raw_chunk)
                if asyncio.iscoroutine(result):
                    result = await result
                metadata = result
            except:
                pass

        content = raw_chunk
        is_last = False

        # Handle SSE format (simplified)
        if "data: " in raw_chunk:
            for line in raw_chunk.split("\n"):
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        is_last = True
                        metadata["finish_reason"] = "done"
                    else:
                        try:
                            parsed = json.loads(data)
                            if "choices" in parsed:
                                if "delta" in parsed["choices"][0]:
                                    content = parsed["choices"][0]["delta"].get("content", "")
                                if "finish_reason" in parsed["choices"][0]:
                                    is_last = True
                                    metadata["finish_reason"] = parsed["choices"][0]["finish_reason"]
                        except:
                            pass

        chunk = StreamChunk(
            content=self._accumulated_content() + content,
            delta=content,
            is_first=self._is_first and content,
            is_last=is_last,
            finish_reason=metadata.get("finish_reason"),
            metadata=metadata
        )

        if content:
            self._is_first = False

        if is_last:
            self._is_done = True

        self.chunks.append(chunk)

        if chunk.delta:
            self._content_buffer.append(chunk.delta)

        # Trigger async callback
        if self.callback:
            result = self.callback(chunk)
            if asyncio.iscoroutine(result):
                await result

    def get_full_content(self) -> str:
        """Get the full accumulated content."""
        return "".join(self._content_buffer)

    def _accumulated_content(self) -> str:
        """Get current accumulated content."""
        return "".join(self._content_buffer)

    @property
    def chunk_count(self) -> int:
        """Get number of chunks received."""
        return len(self.chunks)

    @property
    def estimated_tokens(self) -> int:
        """Get estimated token count."""
        return len(self.get_full_content()) // 4

    @property
    def is_done(self) -> bool:
        """Check if stream is complete."""
        return self._is_done

    def reset(self) -> None:
        """Reset handler state for reuse."""
        self.chunks.clear()
        self._content_buffer.clear()
        self._is_first = True
        self._is_done = False


def stream_response(
    stream: Iterator[str],
    callback: Optional[StreamCallback] = None,
    options: Optional[StreamOptions] = None
) -> str:
    """
    Convenience function to stream a response.

    Args:
        stream: Iterator yielding response chunks
        callback: Optional callback for each chunk
        options: Stream options

    Returns:
        Full accumulated content

    Example:
        def print_chunk(chunk):
            print(chunk.delta, end='', flush=True)

        content = stream_response(response_iterator, callback=print_chunk)
    """
    handler = StreamingHandler(callback=callback, options=options)
    return handler.process_stream(stream)


async def astream_response(
    stream: AsyncIterator[str],
    callback: Optional[AsyncStreamCallback] = None,
    options: Optional[StreamOptions] = None
) -> str:
    """
    Convenience function to stream a response asynchronously.

    Args:
        stream: Async iterator yielding response chunks
        callback: Optional async callback for each chunk
        options: Stream options

    Returns:
        Full accumulated content

    Example:
        async def print_chunk(chunk):
            print(chunk.delta, end='', flush=True)

        content = await astream_response(response_iterator, callback=print_chunk)
    """
    handler = AsyncStreamHandler(callback=callback, options=options)
    return await handler.process_stream(stream)
