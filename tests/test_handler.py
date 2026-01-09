"""
Tests for Streaming Handler
"""

import pytest
import asyncio
from streaming_handler import (
    StreamingHandler,
    AsyncStreamHandler,
    StreamChunk,
    StreamOptions,
    StreamEventType,
    stream_response,
    astream_response,
)
from streaming_handler.parsers import (
    JsonStreamParser,
    parse_partial_json,
    extract_sse_data,
    parse_sse_chunk,
    SSEParser,
)
from streaming_handler.buffer import (
    StreamBuffer,
    ChunkBuffer,
    TokenBuffer,
    BufferedChunk,
    BufferFullError,
)


class TestStreamingHandler:
    """Test StreamingHandler"""

    @pytest.fixture
    def handler(self):
        return StreamingHandler()

    def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler.chunks == []
        assert handler._content_buffer == []
        assert handler._is_first is True
        assert handler._is_done is False

    def test_add_chunk(self, handler):
        """Test adding a chunk"""
        handler.add_chunk("Hello")
        assert handler.get_full_content() == "Hello"
        assert handler.chunk_count == 1

    def test_add_multiple_chunks(self, handler):
        """Test adding multiple chunks"""
        handler.add_chunk("Hello")
        handler.add_chunk(" ")
        handler.add_chunk("world")
        assert handler.get_full_content() == "Hello world"
        assert handler.chunk_count == 3

    def test_add_chunk_with_metadata(self, handler):
        """Test adding chunk with metadata"""
        handler.add_chunk("test", finish_reason="stop")
        assert handler.is_done is True
        assert handler.chunks[-1].finish_reason == "stop"

    def test_reset(self, handler):
        """Test resetting handler"""
        handler.add_chunk("test")
        handler.reset()
        assert handler.get_full_content() == ""
        assert handler.chunk_count == 0
        assert handler._is_first is True

    def test_get_chunks_by_type(self, handler):
        """Test filtering chunks by type"""
        handler.add_chunk("Hello")
        handler.add_chunk("world", finish_reason="stop")

        tokens = handler.get_chunks_by_type(StreamEventType.TOKEN)
        assert len(tokens) == 2

        done = handler.get_chunks_by_type(StreamEventType.DONE)
        assert len(done) == 1

    def test_estimated_tokens(self, handler):
        """Test token estimation"""
        handler.add_chunk("The quick brown fox")
        # ~4 chars per token
        assert handler.estimated_tokens > 0


class TestAsyncStreamHandler:
    """Test AsyncStreamHandler"""

    @pytest.fixture
    def handler(self):
        return AsyncStreamHandler()

    def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler.chunks == []
        assert handler._content_buffer == []

    @pytest.mark.asyncio
    async def test_process_stream(self, handler):
        """Test processing async stream"""
        async def mock_stream():
            yield "Hello"
            yield " "
            yield "world"

        content = await handler.process_stream(mock_stream())
        assert content == "Hello world"
        assert handler.chunk_count == 3

    @pytest.mark.asyncio
    async def test_process_stream_with_done(self, handler):
        """Test processing stream that ends"""
        async def mock_stream():
            yield "Hello"
            yield "[DONE]"

        content = await handler.process_stream(mock_stream())
        assert content == "Hello"

    def test_reset(self, handler):
        """Test resetting handler"""
        handler._content_buffer.append("test")
        handler.reset()
        assert handler.get_full_content() == ""


class TestStreamChunk:
    """Test StreamChunk dataclass"""

    def test_stream_chunk_creation(self):
        """Test creating StreamChunk"""
        chunk = StreamChunk(
            content="Hello",
            delta="Hello",
            is_first=True,
            is_last=False
        )
        assert chunk.content == "Hello"
        assert chunk.is_first is True
        assert chunk.token_count == 1  # 4 chars / 4 = 1


class TestJsonStreamParser:
    """Test JsonStreamParser"""

    @pytest.fixture
    def parser(self):
        return JsonStreamParser()

    def test_parse_complete_json(self, parser):
        """Test parsing complete JSON"""
        result = parser.parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_partial_json(self, parser):
        """Test parsing partial JSON"""
        result = parser.parse('{"key":')
        assert result is None

        # Complete it
        result = parser.parse('"value"}')
        assert result == {"key": "value"}

    def test_parse_nested_json(self, parser):
        """Test parsing nested JSON"""
        result = parser.parse('{"outer": {"inner": "value"}}')
        assert result == {"outer": {"inner": "value"}}

    def test_parse_array(self, parser):
        """Test parsing JSON array"""
        result = parser.parse('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_reset(self, parser):
        """Test resetting parser"""
        parser.parse('{"key":')
        parser.reset()
        assert parser.buffer == ""
        assert parser.bracket_count == 0


class TestParsePartialJson:
    """Test parse_partial_json function"""

    def test_parse_valid_json(self):
        """Test parsing valid JSON"""
        result = parse_partial_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing comma"""
        result = parse_partial_json('{"key": "value",}')
        assert result == {"key": "value"}

    def test_parse_json_from_markdown(self):
        """Test parsing JSON from markdown code block"""
        result = parse_partial_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON"""
        result = parse_partial_json('not json')
        assert result is None


class TestSSEParser:
    """Test SSE parser functions"""

    def test_extract_sse_data(self):
        """Test extracting data from SSE line"""
        data = extract_sse_data("data: {\"test\": true}")
        assert data == '{"test": true}'

    def test_extract_sse_data_done(self):
        """Test extracting [DONE] from SSE"""
        data = extract_sse_data("data: [DONE]")
        assert data == "[DONE]"

    def test_extract_sse_data_invalid(self):
        """Test extracting from invalid SSE line"""
        data = extract_sse_data("not data")
        assert data is None

    def test_parse_sse_chunk(self):
        """Test parsing SSE chunk"""
        result = parse_sse_chunk("data: {\"message\": \"Hello\"}")
        assert result == {"message": "Hello"}

    def test_parse_sse_chunk_done(self):
        """Test parsing SSE done chunk"""
        result = parse_sse_chunk("data: [DONE]")
        assert result is None


class TestSSEParserClass:
    """Test SSEParser class"""

    @pytest.fixture
    def parser(self):
        return SSEParser()

    def test_parse_line(self, parser):
        """Test parsing SSE line"""
        result = parser.parse_line("data: {\"test\": true}\n")
        assert result is not None
        assert result.get("parsed") == {"test": True}

    def test_parse_multiple_lines(self, parser):
        """Test parsing multiple SSE lines"""
        parser.parse_line("data: {\"test\": true}\n")
        parser.parse_line("data: {\"test2\": false}\n")
        assert len(parser.events) == 2

    def test_reset(self, parser):
        """Test resetting parser"""
        parser.parse_line("data: test\n")
        parser.reset()
        assert len(parser.events) == 0
        assert parser.buffer == ""


class TestStreamBuffer:
    """Test StreamBuffer"""

    @pytest.fixture
    def buffer(self):
        return StreamBuffer(max_size=10)

    def test_add_chunk(self, buffer):
        """Test adding chunk"""
        buffer.add("Hello")
        assert buffer.size == 1
        assert buffer.get_content() == "Hello"

    def test_add_multiple_chunks(self, buffer):
        """Test adding multiple chunks"""
        buffer.add("Hello")
        buffer.add(" ")
        buffer.add("world")
        assert buffer.get_content() == "Hello world"
        assert buffer.size == 3

    def test_get_range(self, buffer):
        """Test getting range of chunks"""
        buffer.add("A")
        buffer.add("B")
        buffer.add("C")
        assert buffer.get_content(0, 2) == "AB"
        assert buffer.get_content(1, 3) == "BC"

    def test_clear(self, buffer):
        """Test clearing buffer"""
        buffer.add("test")
        buffer.clear()
        assert buffer.is_empty
        assert buffer.size == 0

    def test_buffer_full_error(self):
        """Test BufferFullError when full"""
        buffer = StreamBuffer(max_size=2, drop_oldest=False)
        buffer.add("a")
        buffer.add("b")

        with pytest.raises(BufferFullError):
            buffer.add("c")

    def test_drop_oldest(self):
        """Test dropping oldest when full"""
        buffer = StreamBuffer(max_size=2, drop_oldest=True)
        buffer.add("a")
        buffer.add("b")
        buffer.add("c")
        assert buffer.size == 2
        assert buffer.get_content() == "bc"


class TestChunkBuffer:
    """Test ChunkBuffer"""

    @pytest.fixture
    def buffer(self):
        return ChunkBuffer(max_chunks=5)

    def test_add_and_get(self, buffer):
        """Test adding and getting chunks"""
        buffer.add("Hello")
        buffer.add(" ")
        buffer.add("world")
        assert buffer.get_content() == "Hello world"

    def test_rolling_behavior(self, buffer):
        """Test rolling buffer drops old chunks"""
        for i in range(10):
            buffer.add(f"{i}")

        assert buffer.size == 5
        assert buffer.get_content() == "56789"

    def test_get_last(self, buffer):
        """Test getting last N chunks"""
        buffer.add("a")
        buffer.add("b")
        buffer.add("c")
        assert buffer.get_last(2) == ["b", "c"]

    def test_clear(self, buffer):
        """Test clearing buffer"""
        buffer.add("test")
        buffer.clear()
        assert buffer.size == 0


class TestTokenBuffer:
    """Test TokenBuffer"""

    @pytest.fixture
    def buffer(self):
        return TokenBuffer(tokens_per_chunk=5)

    def test_add_returns_false_when_not_full(self, buffer):
        """Test add returns False when buffer not full"""
        result = buffer.add("token")
        assert result is False
        assert buffer.size == 1

    def test_add_returns_true_when_full(self, buffer):
        """Test add returns True when buffer is full"""
        for i in range(5):
            assert buffer.add(f"t{i}") == (i == 4)

        assert buffer.size == 5

    def test_get_chunk_clears_buffer(self, buffer):
        """Test get_chunk clears buffer"""
        buffer.add("a")
        buffer.add("b")
        chunk = buffer.get_chunk()
        assert chunk == "ab"
        assert buffer.size == 0

    def test_flush(self, buffer):
        """Test flushing buffer"""
        buffer.add("a")
        buffer.add("b")
        result = buffer.flush()
        assert result == "ab"
        assert buffer.flush() is None

    def test_has_content(self, buffer):
        """Test has_content"""
        assert not buffer.has_content
        buffer.add("test")
        assert buffer.has_content


class TestStreamOptions:
    """Test StreamOptions"""

    def test_default_options(self):
        """Test default options"""
        options = StreamOptions()
        assert options.buffer_size == 100
        assert options.include_metadata is True
        assert options.aggregate_tokens is False

    def test_custom_options(self):
        """Test custom options"""
        options = StreamOptions(
            buffer_size=50,
            include_metadata=False,
            aggregate_tokens=True
        )
        assert options.buffer_size == 50
        assert options.include_metadata is False
        assert options.aggregate_tokens is True


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_stream_response(self):
        """Test stream_response function"""
        chunks = ["Hello", " ", "world"]
        content = stream_response(iter(chunks))
        assert content == "Hello world"

    def test_stream_response_with_callback(self):
        """Test stream_response with callback"""
        received = []

        def callback(chunk):
            received.append(chunk.delta)

        chunks = ["Hello", " ", "world"]
        content = stream_response(iter(chunks), callback=callback)
        assert content == "Hello world"
        assert received == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_astream_response(self):
        """Test astream_response function"""
        async def mock_stream():
            yield "Hello"
            yield " "
            yield "world"

        content = await astream_response(mock_stream())
        assert content == "Hello world"
