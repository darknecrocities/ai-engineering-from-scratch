# Procesamiento de audio en tiempo real

> Los oleoductos de lote procesan un archivo. Los oleoductos en tiempo real procesan los próximos 20 milisegundos antes de que lleguen los próximos 20. Cada IA conversacional, estudio de transmisión y bot de telefonía vive y muere con este presupuesto de latencia.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## El problema

Si quieres un asistente de voz que se sienta vivo, la latencia de toma de vueltas de conversación humana es de ~ 230 ms. Cualquier cosa por encima de 500 ms se siente robótica, más de 1500 ms se siente rota. El presupuesto para una conversación completa**hear → understand → respond → speak**el ciclo en 2026 es:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) logró 200 ms de doble completo. GPT-4o en tiempo real (2024) relojes ~ 320 ms. Las tuberías cascadas en 2022 se enviaron a 2500 ms. La mejora de 10 veces vino de tres técnicas: (1) streaming en todas partes, (2) tuberías asincronas con resultados parciales, (3) generación interrumpida.

## El concepto

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**El audio fluye en tiempo real como bloques de tamaño fijo.

**Ring buffer.**Buffer circular de tamaño fijo. El hilo de productor escribe nuevos marcos, el hilo de consumo lee. Previene las asignaciones en el camino caliente. Tamaño ≈ latencia máxima × tasa de muestra; un anillo de 2 segundos de 16 kHz = 32.000 muestras.

**VAD (Voice Activity Detection).**Los Gates funcionan en aguas subterráneas cuando nadie está hablando. Silero VAD 4.0 (2024) ejecuta <1 ms por 30 ms de fotogramas en CPU. `webrtcvad`es la alternativa más antigua.

**Streaming ASR.**Modelos que emiten transcripciones parciales a medida que llega el audio. Parakeet-CTC-0.6B en modo de transmisión (NeMo, 2024) hace un WER del 25% a una latencia de 320 ms.

**Interruption.**Cuando el usuario habla mientras el asistente habla, debe (a) detectar el barge-in, (b) detener el TTS, (c) descartar la salida restante de LLM. Todo dentro de 100 ms, o el usuario percibe el asistente sordo.

**WebRTC Opus transport.**20 ms de fotogramas, 48 kHz, bitrate adaptativo 8128 kbps. estándar para navegador y móviles. LiveKit, Daily.co, Pion son las pilas 2026 para la construcción de aplicaciones de voz.

**Jitter buffer.**Los paquetes de red llegan fuera de orden / tarde. El buffer jitter reordena y se suaviza; tropiezos → brechas audibles, demasiado grandes → latencia. 6080 ms típicos.

### Gotas comunes

- **Thread contention.**Los modelos pesados GIL + de Python pueden deshacerse del hilo de audio. Utilice una biblioteca de audio de llamada C (dispositivo de sonido, PortAudio) y mantenga a Python fuera del camino caliente.
- **Sample-rate conversion latency.**El nuevo muestreo dentro de la tubería agrega 520 ms. O sea que se vuelva a muestrar de antemano o se utiliza un nuevo muestreo de latencia cero (PolyPhase, `soxr_hq`¿Qué es lo que se hace?
- **TTS priming.**Incluso TTS rápido como Kokoro tiene un calentamiento de 100 200 ms a primera solicitud.
- **Echo cancellation.**Sin AEC, la salida TTS vuelve a entrar en el micrófono y activa ASR en la propia voz del bot.

```figure
nyquist-aliasing
```

## Construye el mismo

### Paso 1: amortiguador de anillos

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

La capacidad determina la latencia máxima de amortiguación. 32.000 muestras a 16 kHz = 2 segundos.

### Paso 2: Puerta de VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Reemplazar con Silero VAD en producción:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Paso 3: transmisión de ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Paso 4: manipulador de interrupciones

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Se puede ver en la transmisión de audio sincronizada y cancelable.

## Usalo

La pila de 2026:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Las trampas

- **Buffering 500 ms to be safe.**El amortiguador es tu piso de latencia.
- **Not pinning threads.**Recall de audio en un hilo de prioridad inferior a la UI = fallos bajo carga.
- **TTS chunks too small.**Los fragmentos de sub-200 ms hacen audibles los artefactos del vocoder.
- **No jitter buffer.**Las redes reales son nerviosas; sin suavizar se obtienen pops.
- **Single-shot error handling.**Las tuberías de audio deben ser a prueba de choque.

## Envío

Salvo como`outputs/skill-realtime-designer.md`Diseñar una línea de audio en tiempo real con presupuestos concretos de latencia por etapa.

## Los ejercicios

1. **Easy.**- ¿ Qué ?`code/main.py`Simula un amortiguador de anillo + energía VAD; Imprime las latencias de etapa para una corriente falsa de 10 segundos.
2. **Medium.**Usando`sounddevice`, construir un paso a través de un bucle que procesa su micrófono en 20 ms de marcos y impresiones estado VAD en cada marco.
3. **Hard.**Construir una prueba de eco duplex completa con `aiortc`: navegador → WebRTC → Python → WebRTC → navegador. Medir la latencia de vidrio a vidrio con un pulso de 1 kHz.

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## Leer más

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) Chunked casi fluyendo susurro.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) 200 ms de latencia de doble completo.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) Orquestación de agentes de producción de audio.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) cancelación de eco bajo código abierto.
