"""
Buffer management for streaming responses.
"""

from typing import List, Dict, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BufferedChunk:
    """A chunk with buffer metadata"""
    content: str
    index: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamBuffer:
    """
    Buffer for managing streaming response chunks.

    Provides configurable buffering strategies for streaming responses.

    Example:
        buffer = StreamBuffer(max_size=1000)

        # Add chunks
        buffer.add("Hello")
        buffer.add(" world")
        buffer.add("!")

        # Get content
        print(buffer.get_content())  # "Hello world!"
        print(buffer.get_chunks())    # List of all chunks
    """

    def __init__(
        self,
        max_size: int = 10000,
        drop_oldest: bool = False
    ):
        """
        Initialize the stream buffer.

        Args:
            max_size: Maximum number of chunks to buffer
            drop_oldest: Whether to drop oldest chunks when full
        """
        self.max_size = max_size
        self.drop_oldest = drop_oldest
        self.chunks: List[BufferedChunk] = []
        self._index = 0

    def add(self, content: str, **metadata) -> None:
        """
        Add a chunk to the buffer.

        Args:
            content: Chunk content
            **metadata: Optional metadata
        """
        chunk = BufferedChunk(
            content=content,
            index=self._index,
            timestamp=datetime.now().timestamp(),
            metadata=metadata
        )
        self._index += 1

        if len(self.chunks) >= self.max_size:
            if self.drop_oldest:
                self.chunks.pop(0)
            else:
                raise BufferFullError(f"Buffer full (max_size={self.max_size})")

        self.chunks.append(chunk)

    def get_content(self, start: int = 0, end: Optional[int] = None) -> str:
        """
        Get buffered content as a string.

        Args:
            start: Starting chunk index
            end: Ending chunk index (exclusive)

        Returns:
            Concatenated content
        """
        chunks = self.chunks[start:end]
        return "".join(c.content for c in chunks)

    def get_chunks(self, start: int = 0, end: Optional[int] = None) -> List[BufferedChunk]:
        """Get chunks as a list."""
        return self.chunks[start:end]

    def get_last(self, n: int = 1) -> str:
        """Get the last n chunks concatenated."""
        return self.get_content(max(0, len(self.chunks) - n), None)

    def clear(self) -> None:
        """Clear all buffered chunks."""
        self.chunks.clear()
        self._index = 0

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.chunks) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.chunks) >= self.max_size

    def get_metadata(self, key: str) -> List[Any]:
        """Get all metadata values for a key."""
        return [c.metadata.get(key) for c in self.chunks if key in c.metadata]


class ChunkBuffer:
    """
    Rolling buffer that keeps only the most recent N chunks.

    Example:
        buffer = ChunkBuffer(max_chunks=100)

        for i in range(1000):
            buffer.add(f"chunk{i}")

        print(buffer.size)  # 100 (keeps last 100)
    """

    def __init__(self, max_chunks: int = 100):
        """
        Initialize the rolling buffer.

        Args:
            max_chunks: Maximum number of chunks to keep
        """
        self.max_chunks = max_chunks
        self._deque: deque = deque(maxlen=max_chunks)
        self._full_content: List[str] = []
        self._dropped_chunks = 0

    def add(self, chunk: str) -> None:
        """
        Add a chunk to the buffer.

        Args:
            chunk: Chunk content
        """
        self._deque.append(chunk)

        # Track dropped chunks for debugging
        if len(self._deque) == self.max_chunks:
            self._dropped_chunks += 1

    def get_content(self) -> str:
        """Get all buffered content concatenated."""
        return "".join(self._deque)

    def get_chunks(self) -> List[str]:
        """Get all chunks as a list."""
        return list(self._deque)

    def get_last(self, n: int) -> List[str]:
        """Get the last n chunks."""
        chunks = list(self._deque)
        return chunks[-n:] if n < len(chunks) else chunks

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self._deque)

    @property
    def dropped_count(self) -> int:
        """Get number of chunks dropped."""
        return self._dropped_chunks

    def clear(self) -> None:
        """Clear the buffer."""
        self._deque.clear()
        self._dropped_chunks = 0


class TokenBuffer:
    """
    Buffer that aggregates tokens before yielding.

    Example:
        buffer = TokenBuffer(tokens_per_chunk=10)

        # Add tokens
        for token in ["Hello", " world", "!", " How", " are", " you?"]:
            if buffer.add(token):
                yield buffer.get_chunk()  # Yields when buffer is full

        # Flush remaining
        if buffer.has_content():
            yield buffer.get_chunk()
    """

    def __init__(self, tokens_per_chunk: int = 10):
        """
        Initialize the token buffer.

        Args:
            tokens_per_chunk: Number of tokens to aggregate before yielding
        """
        self.tokens_per_chunk = tokens_per_chunk
        self._buffer: List[str] = []

    def add(self, token: str) -> bool:
        """
        Add a token to the buffer.

        Args:
            token: Token to add

        Returns:
            True if buffer is full and ready to yield
        """
        self._buffer.append(token)
        return len(self._buffer) >= self.tokens_per_chunk

    def get_chunk(self) -> str:
        """Get and clear buffered tokens."""
        chunk = "".join(self._buffer)
        self._buffer.clear()
        return chunk

    def flush(self) -> Optional[str]:
        """Flush any remaining tokens."""
        if self._buffer:
            return self.get_chunk()
        return None

    @property
    def has_content(self) -> bool:
        """Check if buffer has content."""
        return len(self._buffer) > 0

    @property
    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)


class BufferFullError(Exception):
    """Raised when buffer is full and drop_oldest is False."""
    pass
