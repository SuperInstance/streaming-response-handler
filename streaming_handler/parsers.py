"""
Parsers for streaming responses.
"""

import json
import re
from typing import Any, Dict, Optional, List


class JsonStreamParser:
    """
    Parse JSON from streaming responses.

    Handles incomplete JSON by buffering and parsing when complete.
    Useful for streaming structured responses from LLMs.

    Example:
        parser = JsonStreamParser()

        for chunk in stream:
            result = parser.parse(chunk)
            if result:
                print(f"Parsed: {result}")
    """

    def __init__(self):
        """Initialize the JSON stream parser."""
        self.buffer = ""
        self.bracket_count = 0
        self.in_string = False
        self.escape_next = False

    def parse(self, chunk: str) -> Optional[Any]:
        """
        Parse JSON from a stream chunk.

        Args:
            chunk: JSON fragment from stream

        Returns:
            Parsed object if JSON is complete, None otherwise
        """
        self.buffer += chunk

        # Track bracket/brace balance
        for char in chunk:
            if self.escape_next:
                self.escape_next = False
                continue

            if char == '\\' and self.in_string:
                self.escape_next = True
                continue

            if char == '"' and not self.escape_next:
                self.in_string = not self.in_string
                continue

            if not self.in_string:
                if char in '{[':
                    self.bracket_count += 1
                elif char in '}]':
                    self.bracket_count -= 1

        # Check if we have balanced JSON
        if self.bracket_count == 0 and self.buffer.strip():
            try:
                result = json.loads(self.buffer)
                self.buffer = ""
                return result
            except json.JSONDecodeError:
                pass

        return None

    def reset(self) -> None:
        """Reset the parser state."""
        self.buffer = ""
        self.bracket_count = 0
        self.in_string = False
        self.escape_next = False


def parse_partial_json(json_str: str) -> Optional[Any]:
    """
    Attempt to parse partial/invalid JSON.

    Tries multiple strategies to extract valid JSON from
    incomplete or malformed JSON strings.

    Args:
        json_str: Potentially incomplete JSON string

    Returns:
        Parsed object or None if parsing fails
    """
    # Try direct parse first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', json_str, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting JSON between first { and last }
    brace_match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try extracting array
    array_match = re.search(r'\[.*\]', json_str, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try fixing trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


def extract_sse_data(line: str) -> Optional[str]:
    """
    Extract data from Server-Sent Events (SSE) line.

    Args:
        line: SSE line (e.g., "data: {...}")

    Returns:
        Extracted data string or None
    """
    line = line.strip()
    if not line.startswith("data: "):
        return None

    data = line[6:]  # Remove "data: " prefix
    if data == "[DONE]":
        return data

    return data


def parse_sse_chunk(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse an SSE chunk into a dictionary.

    Args:
        line: SSE line

    Returns:
        Parsed dictionary or None
    """
    data = extract_sse_data(line)
    if not data or data == "[DONE]":
        return None

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


class SSEParser:
    """
    Parse Server-Sent Events (SSE) streams.

    Example:
        parser = SSEParser()

        for line in stream:
            event = parser.parse_line(line)
            if event:
                print(f"Event: {event}")
    """

    def __init__(self):
        """Initialize the SSE parser."""
        self.events: List[Dict[str, Any]] = []
        self.buffer = ""

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single SSE line.

        Args:
            line: Raw SSE line

        Returns:
            Parsed event dict or None
        """
        # Handle multi-line events
        if line.strip():
            self.buffer += line + "\n"
        elif self.buffer:
            # Process complete event
            event = self._parse_event(self.buffer)
            self.buffer = ""
            if event:
                self.events.append(event)
            return event

        return None

    def _parse_event(self, event_text: str) -> Optional[Dict[str, Any]]:
        """Parse a complete SSE event block."""
        event: Dict[str, Any] = {
            "data": None,
            "event": None,
            "id": None,
            "retry": None
        }

        for line in event_text.strip().split('\n'):
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    event['done'] = True
                else:
                    # Multiple data lines are combined with \n
                    if event['data'] is None:
                        event['data'] = data
                    else:
                        event['data'] += '\n' + data
            elif line.startswith('event: '):
                event['event'] = line[7:]
            elif line.startswith('id: '):
                event['id'] = line[4:]
            elif line.startswith('retry: '):
                event['retry'] = int(line[7:])

        # Try to parse data as JSON
        if event.get('data'):
            try:
                event['parsed'] = json.loads(event['data'])
            except json.JSONDecodeError:
                pass

        return event

    def reset(self) -> None:
        """Reset the parser."""
        self.events.clear()
        self.buffer = ""
