

from __future__ import annotations

import asyncio
import logging
import struct
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import numpy as np 
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# CONFIGURATION 


MODEL_ID = "maya-research/Veena"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2  # 16-bit
CHUNK_DURATION_MS = 100
MAX_CONCURRENT_REQUESTS = 5  
DEFAULT_SPEAKER = "kavya"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9

# chunk size
SAMPLES_PER_CHUNK = (SAMPLE_RATE * CHUNK_DURATION_MS) // 1000
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE * NUM_CHANNELS

# Veena control tokens
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

# Available speakers
SPEAKERS = ["kavya", "agastya", "maitri", "vinaya"]


# LOGGING


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tts_server")

# GLOBAL 


model = None
tokenizer = None
snac_model = None
inference_semaphore = None

# No GPU lock needed - TTS runs alone on dedicated G4DN machine
# from gpu_lock import CrossProcessGPUSemaphore
# gpu_lock_manager = None


# MODELS


class TTSRequest(BaseModel):
    
    text: str
    speaker: str = DEFAULT_SPEAKER
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


class HealthResponse(BaseModel):
    
    status: str
    model: str
    speakers: list[str]
    sample_rate: int
    max_concurrent: int
    gpu_available: bool



# MODEL LOADING


def load_models():
    
    global model, tokenizer, snac_model
    
    logger.info("Loading Veena TTS model...")
    start_time = time.time()
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
       
    )
   
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    
    
    logger.info("Loading SNAC decoder...")
    snac_model = SNAC.from_pretrained(SNAC_MODEL_ID).eval().cuda()
    
    load_time = time.time() - start_time
    logger.info(f"Models loaded in {load_time:.2f}s")


# CORE TTS FUNCTIONS

def generate_audio_tokens(text: str, speaker: str, temperature: float, top_p: float) -> list[int]:
   
    global model, tokenizer
    
    # Validate speaker
    if speaker not in SPEAKERS:
        speaker = DEFAULT_SPEAKER
        logger.warning(f"Invalid speaker, using default: {speaker}")
    
    # Prepare input with speaker token
    prompt = f"<spk_{speaker}> {text}"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Construct input sequence
    input_tokens = [
        START_OF_HUMAN_TOKEN,
        *prompt_tokens,
        END_OF_HUMAN_TOKEN,
        START_OF_AI_TOKEN,
        START_OF_SPEECH_TOKEN
    ]
    input_ids = torch.tensor([input_tokens], device=model.device)
    
    # Calculate max tokens based on text length
    max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)
    
    # Generate audio tokens
    with torch.no_grad(), torch.cuda.amp.autocast():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN]
        )
    
    # Extract SNAC tokens
    generated_ids = output[0][len(input_tokens):].tolist()
    snac_tokens = [
        token_id for token_id in generated_ids
        if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
    ]
    
    return snac_tokens


def decode_snac_tokens(snac_tokens: list[int]) -> bytes:
    
    global snac_model
    
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        raise ValueError("Invalid SNAC tokens")
    
    snac_device = next(snac_model.parameters()).device
    llm_codebook_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
    
    # De-interleave tokens into 3 hierarchical levels
    codes_lvl = [[] for _ in range(3)]
    
    for i in range(0, len(snac_tokens), 7):
      
        codes_lvl[0].append(snac_tokens[i] - llm_codebook_offsets[0])
        
        codes_lvl[1].append(snac_tokens[i+1] - llm_codebook_offsets[1])
        codes_lvl[1].append(snac_tokens[i+4] - llm_codebook_offsets[4])
        
        codes_lvl[2].append(snac_tokens[i+2] - llm_codebook_offsets[2])
        codes_lvl[2].append(snac_tokens[i+3] - llm_codebook_offsets[3])
        codes_lvl[2].append(snac_tokens[i+5] - llm_codebook_offsets[5])
        codes_lvl[2].append(snac_tokens[i+6] - llm_codebook_offsets[6])
    
    # Convert to tensors
    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=snac_device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            raise ValueError("Invalid SNAC token values")
        hierarchical_codes.append(tensor)
    
    
    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)
    
    
    audio_np = audio_hat.squeeze().clamp(-1, 1).cpu().numpy()
    pcm_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    pcm_data = pcm_int16.tobytes()
    
    return pcm_data


async def synthesize_speech(text: str, speaker: str, temperature: float, top_p: float) -> bytes:
    
    snac_tokens = generate_audio_tokens(text, speaker, temperature, top_p)
    
    if not snac_tokens:
        raise ValueError("No audio tokens generated")
    
    pcm_data = decode_snac_tokens(snac_tokens)
    
    return pcm_data


async def stream_audio_chunks(
    audio_data: bytes,
    chunk_size: int = BYTES_PER_CHUNK,
) -> AsyncGenerator[bytes, None]:
    """Yield audio data in chunks for streaming response."""
    for i in range(0, len(audio_data), chunk_size):
        yield audio_data[i : i + chunk_size]
        await asyncio.sleep(0)


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global inference_semaphore
    
    # Startup
    load_models()
    
    # HTTP request semaphore (controls queue depth)
    inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    logger.info("TTS running on dedicated GPU (G4DN.XLARGE T4 16GB)")
    
    # Warmup model with test synthesis
    logger.info("Warming up model with test synthesis...")
    try:
        warmup_tokens = generate_audio_tokens("Test", "kavya", 0.4, 0.9)
        warmup_audio = decode_snac_tokens(warmup_tokens)
        logger.info(f"✅ Warmup complete! Generated {len(warmup_audio)} bytes")
    except Exception as e:
        logger.warning(f"⚠️  Warmup failed: {e}")
    
    logger.info(f"Server ready. Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Veena TTS Server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/tts/synthesize")
async def synthesize(request: TTSRequest) -> StreamingResponse:
    """
    Synthesize speech from text.
    
    Returns streaming PCM audio (16-bit signed LE, mono, 24kHz).
    Limited to 10 concurrent requests.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    text_preview = request.text[:50] + "..." if len(request.text) > 50 else request.text
    logger.info(f"Synthesizing [{request.speaker}]: '{text_preview}'")
    
    start_time = time.time()
    
    try:
        async with inference_semaphore:
            pcm_data = await synthesize_speech(
                text=request.text,
                speaker=request.speaker,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        
        inference_time = time.time() - start_time
        audio_duration = len(pcm_data) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        
        logger.info(
            f"Generated {len(pcm_data)} bytes (~{audio_duration:.2f}s audio) "
            f"in {inference_time:.2f}s [RTF: {inference_time/audio_duration:.2f}]"
        )
        
        return StreamingResponse(
            stream_audio_chunks(pcm_data),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Channels": str(NUM_CHANNELS),
                "X-Bit-Depth": "16",
                "X-Audio-Duration": f"{audio_duration:.2f}",
                "X-Inference-Time": f"{inference_time:.2f}",
            },
        )
        
    except ValueError as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="TTS generation failed")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model=MODEL_ID,
        speakers=SPEAKERS,
        sample_rate=SAMPLE_RATE,
        max_concurrent=MAX_CONCURRENT_REQUESTS,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Veena TTS Server",
        "version": "1.0.0",
        "docs": "/docs",
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
