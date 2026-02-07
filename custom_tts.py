"""
Custom TTS Plugin for LiveKit Agents

Connects to external FastAPI TTS service (Veena on RunPod).
Production-ready with connection pooling and proper error handling.

Audio Format:
    - Sample Rate: 24000 Hz
    - Channels: 1 (mono)
    - Bit Depth: 16-bit signed PCM (little-endian)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from livekit.agents.tts import TTS, ChunkedStream, TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

logger = logging.getLogger("custom_tts")


@dataclass
class CustomTTSOptions:
    """Configuration options for CustomTTS."""

    base_url: str
    """Base URL of the TTS service (e.g., http://your-runpod:8001)"""
    sample_rate: int = 24000
    """Audio sample rate in Hz"""
    num_channels: int = 1
    """Number of audio channels (mono)"""
    timeout: float = 60.0
    """HTTP request timeout in seconds (increased for GPU inference)"""
    speaker: str = "kavya"
    """Speaker voice (kavya, agastya, maitri, vinaya)"""


class CustomTTS(TTS):
    """
    Custom TTS plugin for LiveKit that calls an external Veena TTS service.
    
    Designed for production banking vKYC with:
    - Full text synthesis (smooth, natural speech)
    - Connection pooling for efficiency
    - Proper error handling and logging
    """

    def __init__(
        self,
        *,
        base_url: str,
        sample_rate: int = 24000,
        num_channels: int = 1,
        timeout: float = 60.0,
        speaker: str = "kavya",
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._opts = CustomTTSOptions(
            base_url=base_url.rstrip("/"),
            sample_rate=sample_rate,
            num_channels=num_channels,
            timeout=timeout,
            speaker=speaker,
        )
        # Connection pool for handling multiple concurrent requests
        self._http_client: httpx.AsyncClient | None = None
        logger.info(
            f"CustomTTS initialized: base_url={base_url}, "
            f"speaker={speaker}, timeout={timeout}s"
        )

    @property
    def model(self) -> str:
        return "veena-tts"

    @property
    def provider(self) -> str:
        return "maya-research"

    def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create HTTP client with connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self._opts.base_url,
                timeout=httpx.Timeout(self._opts.timeout),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._http_client

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """
        Synthesize text to speech.
        
        Sends full text to TTS server, streams audio back.
        Gives smooth, natural speech output.
        """
        logger.debug(f"synthesize() called: {text[:50]}...")
        return CustomChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            http_client=self._ensure_client(),
            opts=self._opts,
        )

    async def aclose(self) -> None:
        """Close the HTTP client when done."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
            logger.debug("HTTP client closed")


class CustomChunkedStream(ChunkedStream):
    """
    Streams audio chunks from the TTS server.
    
    Receives full text, streams back PCM audio chunks.
    """

    def __init__(
        self,
        *,
        tts: CustomTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        http_client: httpx.AsyncClient,
        opts: CustomTTSOptions,
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._http_client = http_client
        self._opts = opts

    async def _run(self, output_emitter) -> None:
        """Stream audio from TTS server."""
        request_id = shortuuid()
        text_preview = self._input_text[:50] + "..." if len(self._input_text) > 50 else self._input_text
        logger.info(f"[{request_id}] TTS request: {text_preview}")

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            mime_type="audio/pcm",
        )

        try:
            async with self._http_client.stream(
                "POST",
                "/tts/synthesize",
                json={
                    "text": self._input_text,
                    "speaker": self._opts.speaker,
                },
            ) as response:
                response.raise_for_status()

                bytes_received = 0
                # 100ms chunks at 24kHz 16-bit mono = 4800 bytes
                async for chunk in response.aiter_bytes(chunk_size=4800):
                    output_emitter.push(chunk)
                    bytes_received += len(chunk)

            output_emitter.flush()
            
            audio_duration = bytes_received / (self._opts.sample_rate * 2)
            logger.info(f"[{request_id}] Complete: {bytes_received} bytes (~{audio_duration:.2f}s)")

        except httpx.HTTPStatusError as e:
            logger.error(f"[{request_id}] HTTP {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"[{request_id}] Request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}")
            raise
