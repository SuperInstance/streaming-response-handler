# Streaming Response Handler

Handle streaming LLM responses elegantly with buffering, callbacks, and SSE parsing.

## Features

- **Unified Interface**: Single API for all LLM providers
- **Buffer Management**: Configurable buffering strategies
- **Callback Support**: Sync and async callbacks for each chunk
- **SSE Parsing**: Built-in Server-Sent Events parser
- **JSON Streaming**: Parse partial JSON from streams
- **Token Estimation**: Track tokens during streaming
- **Async Support**: Full async/await support

## Installation

```bash
pip install streaming-response-handler
```

## Quick Start

### Basic Streaming

```python
from streaming_handler import stream_response, StreamingHandler

# With callback
def on_chunk(chunk):
    print(chunk.delta, end='', flush=True)

content = stream_response(response_iterator, callback=on_chunk)

# Or with handler
handler = StreamingHandler()
handler.process_stream(response_iterator)
print(f"\nFull: {handler.get_full_content()}")
```

### Async Streaming

```python
from streaming_handler import astream_response, AsyncStreamHandler

async def main():
    # Simple version
    content = await astream_response(async_iterator, callback=on_chunk)

    # Or with handler
    handler = AsyncStreamHandler()
    await handler.process_stream(async_iterator)
```

### SSE Parsing

```python
from streaming_handler import SSEParser

parser = SSEParser()
for line in stream:
    event = parser.parse_line(line)
    if event and event.get('parsed'):
        print(f"Got: {event['parsed']}")
```

### JSON Streaming

```python
from streaming_handler import JsonStreamParser

parser = JsonStreamParser()
for chunk in stream:
    result = parser.parse(chunk)
    if result:
        print(f"Complete JSON: {result}")
```

## API Reference

### StreamingHandler

```python
handler = StreamingHandler(callback=optional_callback, options=optional_options)

# Process a stream
handler.process_stream(stream, metadata_parser=None)

# Add chunks manually
handler.add_chunk(content, finish_reason=None)

# Get results
handler.get_full_content()
handler.chunk_count
handler.estimated_tokens
handler.is_done
```

### AsyncStreamHandler

```python
handler = AsyncStreamHandler(callback=optional_async_callback, options=optional_options)

# Process async stream
await handler.process_stream(stream, metadata_parser=None)

# Get results (same as sync version)
handler.get_full_content()
handler.chunk_count
handler.is_done
```

### StreamChunk

```python
@dataclass
class StreamChunk:
    content: str           # Accumulated content
    delta: str             # New content in this chunk
    is_first: bool         # First chunk
    is_last: bool          # Last chunk
    finish_reason: Optional[str]
    metadata: Dict[str, Any]
```

### Buffers

```python
from streaming_handler import StreamBuffer, ChunkBuffer, TokenBuffer

# Fixed-size buffer
buffer = StreamBuffer(max_size=1000)
buffer.add("chunk")
buffer.get_content()

# Rolling buffer (keeps last N chunks)
rolling = ChunkBuffer(max_chunks=100)
rolling.add("chunk")
rolling.get_content()

# Token aggregation buffer
token_buf = TokenBuffer(tokens_per_chunk=10)
if token_buf.add(token):
    yield token_buf.get_chunk()
```

## License

MIT License - see LICENSE file for details.
