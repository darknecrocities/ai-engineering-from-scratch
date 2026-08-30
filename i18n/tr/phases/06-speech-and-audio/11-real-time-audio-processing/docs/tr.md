# Gerçek Zamanlı Ses İşleme

> Batch boru hattları bir dosyayı işliyor. Gerçek zamanlı boru hattları, sonraki 20 milisaniye gelmeden sonraki 20 milisaniyeyi işliyor. Her konuşma yapay zeka, yayın stüdyosu ve telefon robotu bu gecikme bütçesiyle yaşar ve ölür.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## Sorun

İnsan konuşmalarının dönüş süresi 230 ms'dir. 500 ms'den fazla olan her şey robot gibi hisseder. 1500 ms'den fazla olan ise kırılmış gibi hisseder.**hear → understand → respond → speak**2026'da döngü:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) 200 ms tam çiftlik saatini yaptı. GPT-4o gerçek zamanlı saatler (2024) ~ 320 ms. 2022'de kaskad boru hattları 2500 ms'de gönderildi. 10x iyileştirme üç teknikten geldi: (1) her yerde akış, (2) kısmi sonuçlarla asinkron boru hattı, (3) kesilebilir üretim.

## Anlaşım

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**Gerçek zamanlı ses akışı sabit boyutlu bloklar olarak. Genel seçim: 20 ms (320 örnek 16 kHz).

**Ring buffer.**Sıkı boyutlu döngü tampon. Üreticiler fişi yeni çerçeveler yazar, tüketici fişi okuyor. Sıcak yolda tahsis edilmesini engeller. Ölçüm ≈ maksimum gecikme × örnek hızı; 2 saniye 16 kHz yüzüğü = 32.000 örnek.

**VAD (Voice Activity Detection).**Silero VAD 4.0 (2024) CPU'da 30 ms çerçeve başına < 1 ms çalışır. `webrtcvad`Bu daha eski bir alternatif.

**Streaming ASR.**Ses geldiğinde kısmi transkriptleri yayan modeller. Parakeet-CTC-0.6B akış modunda (NeMo, 2024) 320 ms gecikme ile %2 5% WER yapar. Whisper-Streaming (Macháček et al., 2023) ~ 2 saniye gecikme ile yakında akışmak için Whisper parçaları.

**Interruption.**Kullanıcı konuşurken asistan konuşurken, (a) barge-in'i algılamanız, (b) TTS'yi durdurmanız, (c) kalan LLM çıkışını atmanız gerekir.

**WebRTC Opus transport.**20 ms çerçeveleri, 48 kHz, adapte bit hızı 8128 kbps. Tarayıcı ve mobil için standart. LiveKit, Daily.co, Pion ses uygulamaları oluşturmak için 2026 yığınlarıdır.

**Jitter buffer.**Ağ paketleri düzensiz / geç gelir. Jitter tamponu yeniden düzenlenir ve düzeltir; çok küçük → sesli boşluklar, çok büyük → gecikme. 6080 ms tipik.

### Ortak çöpler

- **Thread contention.**Python'un GIL + ağır modelleri ses düğümünü aç bırakabilir. C- çağrı geri ses kütüphanesi (ses cihazı, PortAudio) kullanın ve Python'u sıcak yoldan uzak tutun.
- **Sample-rate conversion latency.**Boru hattının içinde yeniden örnekleme 520 ms ekler. Ya önce yeniden örnekleme veya sıfır gecikme yeniden örnekleme kullanın (PolyPhase, `soxr_hq`)
- **TTS priming.**Kokoro gibi hızlı TTS'ler bile ilk istek üzerine 100 200 ms ısınma süresi kullanıyor.
- **Echo cancellation.**AEC olmadan, TTS çıkışı mikrofona geri giriyor ve botun kendi sesinde ASR'yi tetikler. WebRTC AEC3 açık kaynaklı varsayılanıdır.

```figure
nyquist-aliasing
```

## Yapın

### Adım 1: Yüzük tamponu

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

Kapasite maksimum tampon gecikmesini belirler. 32 bin örnek 16 kHz = 2 saniye.

### Adım 2: VAD kapısı

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Silero VAD ile değiştirilmek:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Adım 3: Akış ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 4. Adım: Kesinlik yöneticisi

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

Async I/O ve iptal edilebilir TTS akışında. WebRTC peerconnection.stop() ses izinde kanonik yol.

## Kullan

2026'da:

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## Tuzaklar

- **Buffering 500 ms to be safe.**Bufer, gecikme zemininiz.
- **Not pinning threads.**Uayından öncelikli olarak düşük bir düğümde ses çağrısı = yük altında hatalar.
- **TTS chunks too small.**200 ms altındaki parçalar Vocoder eserlerini duyururur. 320 ms parçaları en güzel noktayı.
- **No jitter buffer.**Gerçek ağlar sinirlidir. Yumuşatmadan poplar olur.
- **Single-shot error handling.**Ses boruları çarpışmalara karşı sağlam olmalı.

## Gönder

- Kaydet .`outputs/skill-realtime-designer.md`.Eğer bir aşama için belirli gecikme bütçeleri ile gerçek zamanlı bir ses borusunu tasarlayın.

## Egzersizler

1. **Easy.**Çık .`code/main.py`. Bir yüzük tamponu + enerji VAD simülasyonu; sahte 10 saniye akış için aşama gecikmelerini yazdırır.
2. **Medium.**Kullanım`sounddevice`, 20 ms çerçeve içinde mikrofonunu işleyen ve her çerçeveye VAD durumu yazdırırmak için bir geçiş yolu döngüsü oluşturmak.
3. **Hard.**Tam bir duplex ekro testi yapın `aiortc`: browser → WebRTC → Python → WebRTC → browser. 1 kHz nabızla camdan camya gecikmeyi ölçün.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## Daha Fazla Okumak

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) Yaklaşık akışlı fısıldama.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) Tam duplex 200 ms gecikme.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) üretim ses ajan orkestrasyonu.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) Açık kaynak altında yankon iptal edilmesi.
