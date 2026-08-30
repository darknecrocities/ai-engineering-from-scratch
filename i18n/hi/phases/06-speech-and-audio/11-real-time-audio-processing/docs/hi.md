# वास्तविक समय में ऑडियो प्रोसेसिंग

> बैच पाइपलाइन एक फ़ाइल को संसाधित करती है। अगले 20 मिलिसेंड के आने से पहले वास्तविक समय पाइपलाइन अगले 20 मिलिसेकेंड को संसाधित करती है। हर वार्तालाप एआई, प्रसारण स्टूडियो और टेलीफोन बॉट इस लटेंसी बजट के साथ रहता है और मर जाता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## समस्या

आप एक आवाज सहायक चाहते हैं जो जीवित महसूस करता है। मानव वार्तालाप टर्न-टेक लेटेंसी ~ 230 ms है (चुपचाप-से-जवाब) । 500 ms से ऊपर की कोई भी चीज रोबोटिक लगती है; 1500 ms से ऊपर टूट लगता है। पूर्ण के लिए बजट**hear → understand → respond → speak**2026 में लूप हैः

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

मोशी (क्यूताई, 2024) ने 200 एमएस पूर्ण-डूप्लेक्स घड़ी बनाई। जीपीटी-4o-रियल टाइम (2024) घड़ियों ~ 320 एमएस। 2022 में 2500 एमएस पर कैस्केडेड पाइपलाइनें भेजी गईं। 10x सुधार तीन तकनीकों से आयाः (1) हर जगह स्ट्रीमिंग, (2) आंशिक परिणामों के साथ असिनक्रोनस पाइपलाइन, (3) बाधित पीढ़ी।

## अवधारणा

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**वास्तविक समय में ऑडियो फिक्स्ड साइज ब्लॉक के रूप में प्रवाह करता है। आम विकल्पः 20 ms (320 नमूने 16 kHz पर) । नीचे की ओर सब कुछ इस गति के साथ रहना चाहिए।

**Ring buffer.**फिक्स्ड साइज सर्कुलर बफर। निर्माता धागा नए फ्रेम लिखता है, उपभोक्ता धागा पढ़ता है। हॉट पथ में आवंटन को रोकता है। आकार ≈ अधिकतम विलंबता × नमूना दर; 2-सेकंड 16 kHz रिंग = 32,000 नमूने।

**VAD (Voice Activity Detection).**सिलेरो VAD 4.0 (2024) CPU पर प्रति 30 ms फ्रेम <1 ms चलाता है। `webrtcvad`यह पुराना विकल्प है।

**Streaming ASR.**ऑडियो आने के साथ आंशिक प्रतिलेखन उत्सर्जित करने वाले मॉडल। स्ट्रीमिंग मोड में पैराकीट-सीटीसी-0.6 बी (NeMo, 2024) 320 ms विलंबता पर 25% WER करता है। विस्पर-स्ट्रीमिंग (Macháček et al., 2023) ~ 2s विलंबता पर निकट-स्ट्रीमिंग के लिए विस्पर टुकड़े करता है।

**Interruption.**जब उपयोगकर्ता सहायक बोल रहा है, तो आपको (ए) बार्ज-इन का पता लगाना चाहिए, (बी) टीटीएस को रोकना चाहिए, (सी) शेष एलएलएम आउटपुट को फेंकना चाहिए। सभी 100 एमएस के भीतर, या उपयोगकर्ता बहरे सहायक को महसूस करता है।

**WebRTC Opus transport.**20 एमएस फ्रेम, 48 किलोग्राम हर्ट्ज, 8128 किलोग्राम प्रति सेकंड। ब्राउज़र और मोबाइल के लिए मानक। लाइवकिट, डेली डॉट कॉ, पियान वॉयस ऐप बनाने के लिए 2026 स्टैक हैं।

**Jitter buffer.**नेटवर्क पैकेट आदेश से बाहर / देर से आते हैं। jitter बफर रीऑर्डर और चिकनी; बहुत छोटे → श्रव्य अंतराल, बहुत बड़ा → विलंबता। 6080 ms आम है।

### सामान्य गॉथ

- **Thread contention.**पायथन के जीआईएल + भारी मॉडल ऑडियो थ्रेड को भूख से मार सकते हैं। सी-कॉलबैक ऑडियो लाइब्रेरी (साउंडडिवाइस, पोर्टऑडियो) का उपयोग करें और पायथन को हॉट पथ से दूर रखें।
- **Sample-rate conversion latency.**पाइपलाइन के अंदर पुनः नमूनाकरण 520 ms जोड़ता है। या तो अग्रिम में पुनः नमूनाकरण करें या शून्य विलंबता पुनः नमूनाकरण (पॉलीफेस, `soxr_hq`) ।
- **TTS priming.**यहां तक कि कोकोरो जैसे तेज टीटीएस में पहले अनुरोध पर 100200 एमएस वार्म-अप होता है. कैश मॉडल + पहले वास्तविक वक्र से पहले इसे एक डमी रन के साथ गर्म करता है।
- **Echo cancellation.**एईसी के बिना, टीटीएस आउटपुट माइक्रोफ़ोन में वापस प्रवेश करता है और बॉट की अपनी आवाज पर एएसआर को ट्रिगर करता है। वेबआरटीसी एईसी 3 ओपन-सोर्स डिफ़ॉल्ट है।

```figure
nyquist-aliasing
```

## इसे बनाओ

### चरण 1: रिंग बफर

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

क्षमता अधिकतम बफरिंग विलंबता निर्धारित करता है। 32000 नमूने 16 kHz = 2 सेकंड पर.

### चरण 2: VAD गेट

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

उत्पादन में सिलेरो VAD से प्रतिस्थापित करेंः

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### चरण 3: स्ट्रीमिंग एएसआर

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### चरण 4: विराम के लिए संभाल

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

वेबआरटीसी पीयरकनेक्शन.स्टॉप() ऑडियो ट्रैक पर कैनोनिक तरीका है।

## इसका प्रयोग करें

2026 स्टैकः

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## फंदे

- **Buffering 500 ms to be safe.**बफर *आपकी विलंबता मंजिल है. इसे संकुचित करें.
- **Not pinning threads.**प्राथमिकता-निम्न-UI थ्रेड पर ऑडियो कॉलबैक = लोड के तहत ग्लिच।
- **TTS chunks too small.**200 एमएस के तहत टुकड़े Vocoder कलाकृतियों को श्रव्य बनाते हैं। 320 एमएस टुकड़े सबसे अच्छा बिंदु हैं।
- **No jitter buffer.**वास्तविक नेटवर्क घबराहट में हैं; बिना सुचारू किए आप पॉप प्राप्त करते हैं।
- **Single-shot error handling.**ऑडियो पाइपलाइनों को दुर्घटना-प्रूफ होना चाहिए. एक अपवाद सत्र को मारता है.

## इसे भेजें

`outputs/skill-realtime-designer.md`. प्रत्येक चरण के लिए ठोस विलंबता बजट के साथ वास्तविक समय में ऑडियो पाइपलाइन डिजाइन करें।

## व्यायाम

1. **Easy.**दौड़ें`code/main.py`. रिंग बफर + ऊर्जा VAD का अनुकरण करता है; एक नकली 10 सेकंड धारा के लिए चरण विलंबता प्रिंट करता है।
2. **Medium.**उपयोग करना`sounddevice`, एक पार पार लूप बनाने जो अपने माइक्रोसॉफ्ट को 20 ms फ्रेम में संसाधित करता है और प्रत्येक फ्रेम पर VAD स्थिति प्रिंट करता है।
3. **Hard.** के साथ एक पूर्ण डुप्लेक्स इको परीक्षण का निर्माण करें`aiortc`: ब्राउज़र → वेबआरटीसी → पायथन → वेबआरटीसी → ब्राउज़र. 1 किलोग्राम हर्ट्ज की पल्स के साथ ग्लास-टू-ग्लास विलंबता मापें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## आगे पढ़ना

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) लगभग प्रवाह विस्मयकारी टुकड़े.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) पूर्ण डुप्लेक्स 200 एमएस विलंबता।
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) उत्पादन ऑडियो एजेंट ऑर्केस्ट्रेशन।
- [Silero VAD repo](https://github.com/snakers4/silero-vad) sub-1 ms VAD, Apache 2.0
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) ओपन सोर्स के तहत इको रद्द करना।
