"""
Retry logic with exponential backoff for streaming responses.

Handles transient failures like network errors, rate limits, and timeouts.
"""

import asyncio
import random
import time
from typing import (
    Optional,
    Callable,
    Type,
    Tuple,
    Any,
    AsyncIterator,
    Iterator,
    List,
    Dict,
    Union,
)
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategy types"""
    EXPONENTIAL = "exponential"  # Exponential backoff with jitter
    LINEAR = "linear"  # Linear backoff
    IMMEDIATE = "immediate"  # No delay between retries


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial attempt)
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        strategy: Retry strategy to use
        jitter: Add random jitter to delays (0.0 = none, 1.0 = full jitter)
        retryable_errors: Error types that should trigger a retry
        retryable_status_codes: HTTP status codes that should trigger a retry
        on_retry_callback: Optional callback called before each retry
    """
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: float = 0.1  # Add 10% randomness by default
    retryable_errors: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    retryable_status_codes: Tuple[int, ...] = (
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    )
    on_retry_callback: Optional[Callable[[int, Exception, float], None]] = None

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt.

        Args:
            attempt: Retry attempt number (1-based)

        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay * attempt
        else:  # EXPONENTIAL
            delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))

        # Apply max delay cap
        delay = min(delay, self.max_delay)

        # Add jitter
        if self.jitter > 0 and delay > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def is_retryable(self, error: Exception) -> bool:
        """
        Check if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if the error should trigger a retry
        """
        # Check error type
        if isinstance(error, self.retryable_errors):
            return True

        # Check for HTTP status codes in error message or attributes
        error_str = str(error).lower()
        for code in self.retryable_status_codes:
            if str(code) in error_str:
                return True

        # Check for rate limit indicators
        rate_limit_phrases = ["rate limit", "too many requests", "quota exceeded"]
        return any(phrase in error_str for phrase in rate_limit_phrases)


@dataclass
class RetryResult:
    """Result of a retry operation"""
    success: bool
    attempts: int
    total_delay: float
    last_error: Optional[Exception] = None
    content: str = ""

    @property
    def failed(self) -> bool:
        """Check if the operation failed after all retries"""
        return not self.success


class RetryableStreamIterator:
    """
    Wraps a stream iterator with automatic retry on failure.

    Example:
        def fetch_stream():
            # This might fail due to network issues
            return api.stream(...)

        retry_stream = RetryableStreamIterator(fetch_stream)
        for chunk in retry_stream:
            print(chunk, end='')

        if retry_stream.result.failed:
            print(f"Failed after {retry_stream.result.attempts} attempts")
    """

    def __init__(
        self,
        stream_factory: Callable[[], Iterator[str]],
        config: Optional[RetryConfig] = None,
        state_preserver: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        """
        Initialize the retryable stream iterator.

        Args:
            stream_factory: Function that creates a new stream iterator
            config: Retry configuration
            state_preserver: Optional function to capture state for replay
        """
        self.stream_factory = stream_factory
        self.config = config or RetryConfig()
        self.state_preserver = state_preserver

        self.result = RetryResult(
            success=False,
            attempts=0,
            total_delay=0.0,
        )
        self._current_stream: Optional[Iterator[str]] = None
        self._replay_buffer: List[str] = []
        self._state_snapshot: Optional[Dict[str, Any]] = None

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        while True:
            try:
                # Get next chunk from current stream
                if self._current_stream is None:
                    self._start_new_stream()

                chunk = next(self._current_stream)

                # Store chunk for potential replay
                if self.state_preserver:
                    self._replay_buffer.append(chunk)

                return chunk

            except StopIteration:
                # Stream completed successfully
                self.result.success = True
                raise

            except Exception as e:
                self.result.last_error = e
                self.result.attempts += 1

                # Check if we should retry
                if self.result.attempts < self.config.max_attempts and self.config.is_retryable(e):
                    delay = self.config.get_delay(self.result.attempts)
                    self.result.total_delay += delay

                    # Call retry callback if provided
                    if self.config.on_retry_callback:
                        self.config.on_retry_callback(self.result.attempts, e, delay)

                    logger.warning(
                        f"Stream interrupted (attempt {self.result.attempts}/{self.config.max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

                    # Replay captured chunks if needed
                    if self._replay_buffer and self._state_preserver:
                        yield from self._replay_buffer

                    # Start new stream
                    self._start_new_stream()

                else:
                    # Max attempts exceeded or non-retryable error
                    logger.error(
                        f"Stream failed after {self.result.attempts} attempts: {e}"
                    )
                    raise

    def _start_new_stream(self) -> None:
        """Start a new stream iterator."""
        self._current_stream = self.stream_factory()
        self._replay_buffer.clear()


class AsyncRetryableStreamIterator:
    """
    Async version of RetryableStreamIterator.

    Example:
        async def fetch_stream():
            # This might fail due to network issues
            return await api.astream(...)

        retry_stream = AsyncRetryableStreamIterator(fetch_stream)
        async for chunk in retry_stream:
            print(chunk, end='')
    """

    def __init__(
        self,
        stream_factory: Callable[[], AsyncIterator[str]],
        config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the async retryable stream iterator.

        Args:
            stream_factory: Function that creates a new async stream iterator
            config: Retry configuration
        """
        self.stream_factory = stream_factory
        self.config = config or RetryConfig()

        self.result = RetryResult(
            success=False,
            attempts=0,
            total_delay=0.0,
        )
        self._current_stream: Optional[AsyncIterator[str]] = None

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        while True:
            try:
                # Get next chunk from current stream
                if self._current_stream is None:
                    self._current_stream = self.stream_factory()

                chunk = await self._current_stream.__anext__()
                return chunk

            except StopAsyncIteration:
                # Stream completed successfully
                self.result.success = True
                raise

            except Exception as e:
                self.result.last_error = e
                self.result.attempts += 1

                # Check if we should retry
                if self.result.attempts < self.config.max_attempts and self.config.is_retryable(e):
                    delay = self.config.get_delay(self.result.attempts)
                    self.result.total_delay += delay

                    # Call retry callback if provided
                    if self.config.on_retry_callback:
                        cb_result = self.config.on_retry_callback(self.result.attempts, e, delay)
                        if asyncio.iscoroutine(cb_result):
                            await cb_result

                    logger.warning(
                        f"Stream interrupted (attempt {self.result.attempts}/{self.config.max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

                    # Start new stream
                    self._current_stream = self.stream_factory()

                else:
                    # Max attempts exceeded or non-retryable error
                    logger.error(
                        f"Stream failed after {self.result.attempts} attempts: {e}"
                    )
                    raise


def retry_stream(
    stream_factory: Callable[[], Iterator[str]],
    config: Optional[RetryConfig] = None,
) -> RetryableStreamIterator:
    """
    Wrap a stream factory with retry logic.

    Args:
        stream_factory: Function that creates a new stream iterator
        config: Retry configuration

    Returns:
        A retryable stream iterator

    Example:
        def fetch_stream():
            return requests.get(url, stream=True).iter_lines()

        for chunk in retry_stream(fetch_stream):
            process(chunk)
    """
    return RetryableStreamIterator(stream_factory, config)


def aretry_stream(
    stream_factory: Callable[[], AsyncIterator[str]],
    config: Optional[RetryConfig] = None,
) -> AsyncRetryableStreamIterator:
    """
    Wrap an async stream factory with retry logic.

    Args:
        stream_factory: Function that creates a new async stream iterator
        config: Retry configuration

    Returns:
        An async retryable stream iterator

    Example:
        async def fetch_stream():
            async for chunk in async_client.stream(...):
                yield chunk

        async for chunk in aretry_stream(fetch_stream):
            process(chunk)
    """
    return AsyncRetryableStreamIterator(stream_factory, config)


def retry_decorator(
    config: Optional[RetryConfig] = None,
):
    """
    Decorator to add retry logic to streaming functions.

    Args:
        config: Retry configuration

    Example:
        @retry_decorator(config=RetryConfig(max_attempts=5))
        def fetch_messages(prompt: str):
            return api.stream(prompt)

        for chunk in fetch_messages("hello"):
            print(chunk)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Union[RetryableStreamIterator, AsyncRetryableStreamIterator]:
            stream_factory = lambda: func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                # Async function - needs special handling
                async def async_wrapper():
                    return AsyncRetryableStreamIterator(stream_factory, config)
                return async_wrapper()
            else:
                return RetryableStreamIterator(stream_factory, config)

        return wrapper
    return decorator


# Pre-configured retry strategies for common scenarios

DEFAULT_RETRY = RetryConfig()  # Standard retry

AGGRESSIVE_RETRY = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    backoff_multiplier=1.5,
    jitter=0.2,
)

CONSERVATIVE_RETRY = RetryConfig(
    max_attempts=2,
    initial_delay=2.0,
    backoff_multiplier=3.0,
    jitter=0.05,
)

RATE_LIMIT_RETRY = RetryConfig(
    max_attempts=10,
    initial_delay=5.0,
    max_delay=300.0,  # 5 minutes max
    backoff_multiplier=2.0,
    jitter=0.3,
    retryable_status_codes=(429, 500, 502, 503, 504),
)
