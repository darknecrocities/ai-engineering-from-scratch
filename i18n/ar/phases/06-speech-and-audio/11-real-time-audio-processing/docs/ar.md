# معالجة الصوت في الوقت الحقيقي

> خطوط الأنابيب المكونة من مجموعات معالجة ملف. خطوط الأنابيب في الوقت الحقيقي معالجة 20 مل ثانية القادمة قبل وصول 20 المقبل. كل الذكاء الاصطناعي المحادثات، الاستوديو البث، والروبوتات الهاتفية يعيش وتموت من خلال هذا الميزانية التخفيف.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## المشكلة

تريد مساعد صوتي يشعر بالحياة. تأخير تحويل المحادثة البشرية هو ~ 230 ms (صمت إلى استجابة). أي شيء فوق 500 ms يشعر الروبوتي؛ فوق 1500 ms يشعر الفشل. الميزانية لملء كامل **hear → understand → respond → speak**الحلقة في عام 2026 هي:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

موسشي (كيوتاي ، 2024) حققت 200 ميس كامل المزدوج. ساعات GPT-4o-realtime (2024) ~ 320 ميس. شحن خطوط الأنابيب المتساقطة في 2022 في 2500 ميس. جاء التحسن 10x من ثلاثة تقنيات: (1) التدفق في كل مكان ، (2) التدفق غير المزامن مع نتائج جزئية ، (3) توليد قابل للانقطاع.

## المفهوم

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**تدفق الصوت في الوقت الحقيقي ككتلات ذات الحجم الثابت. الخيار الشائع: 20 ms (320 عينات عند 16 kHz). كل شيء أسفل التيار يجب أن يتوافق مع هذه التدفق.

**Ring buffer.**مسدس دائري بحجم ثابت. يكتب الخيط المنتج إطارات جديدة، يقرأ الخيط الاستهلاكي. يمنع التخصيص في المسار الساخن. الحجم ≈ أقصى تأخر × معدل العينة؛ حلقة 16 كيلوغرتز لمدة 2 ثانية = 32000 عينة.

**VAD (Voice Activity Detection).**يعمل غايتس أسفل التيار عندما لا أحد يتحدث. Silero VAD 4.0 (2024) يعمل < 1 ms لكل إطار 30 ms على CPU. `webrtcvad`هو البديل الأكبر سنا.

**Streaming ASR.**النماذج التي تنبعث من النصوص الجزئية أثناء وصول الصوت. Parakeet-CTC-0.6B في وضع التدفق (NeMo ، 2024) يقدم 25% WER عند تأخر 320 ms. Whisper-Streaming (Macháček et al., 2023) قطع Whisper للتدفق القريب عند تأخر ~ 2 ثانية.

**Interruption.**عندما يتحدث المستخدم بينما يتحدث المساعد، يجب عليك (أ) اكتشاف المبارزة، (ب) إيقاف TTS، (ج) التخلص من الناتج المتبقي لـ LLM. كل ذلك في غضون 100 ms، أو يلاحظ المستخدم المساعد الصم.

**WebRTC Opus transport.**20 ميس إطار، 48 كيلوهرتز، معدل التنفيذ التكيفي 8128 كيلوهرتز. معيار للمتنشر والجوال. LiveKit، Daily.co، Pion هي كومات 2026 لبناء تطبيقات الصوت.

**Jitter buffer.**وصل حزم الشبكة خارج النظام / متأخرة. يعيد نظام التنظيم والسلاسة؛ فجوات صغيرة جدا → صوتية، كبيرة جدا → تأخير. 6080 ms نموذجيا.

### المواد المشتركة

- **Thread contention.**يمكن أن تقوم النماذج الثقيلة GIL + Python بإغلاق الخيط الصوتي. استخدم مكتبة صوتية C-callback (جهاز صوتي، PortAudio) وابق Python بعيدا عن المسار الساخن.
- **Sample-rate conversion latency.**إعادة العينة داخل خط الأنابيب يضيف 520 ms إما إعادة العينة مقدماً أو استخدام إعادة العينة ذات التأخير الصفر (PolyPhase، `soxr_hq`)
- **TTS priming.**حتى TTS سريع مثل كوكورو لديه 100200 ms التدفئة على الطلب الأول. نموذج الاحتياطي + دافئها مع تشغيل وهمية قبل التحول الحقيقي الأول.
- **Echo cancellation.**بدون AEC ، يُعيد إنتاج TTS الدخول إلى الميكروفون ويُؤدي إلى ASR على صوت الروبوت نفسه. ويبرتك AEC3 هو افتتاحي المصدر المفتوح.

```figure
nyquist-aliasing
```

## بناءها

### الخطوة الأولى: خفيفة الحلقة

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

السعة تحدد أقصى وقت تأخير البفرة 32000 عينات عند 16 كيلوهرتز = 2 ثانية

### الخطوة الثانية: بوابة VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

استبدال Silero VAD في الإنتاج:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### الخطوة الثالثة: تسليم ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### الخطوة الرابعة: معالج التقاطع

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

يتعلق على التشغيل / التشغيل غير المزامن وتسريب التسويق القابل للتلغي. WebRTC peerconnection.stop() على المسار الصوتي هو الطريقة القنوني.

## استخدمها

"مجموعة 2026"

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## الفخاخ

- **Buffering 500 ms to be safe.**المكافأة هي سطحك التأخير، قم بتقليصها
- **Not pinning threads.**إعادة الاتصال الصوتي على خيط أقل من الوضع الأولوي = أخطاء تحت الحمل.
- **TTS chunks too small.**قطع تحت 200 ميس يجعل القطع الأثرية الصوتية.
- **No jitter buffer.**الشبكات الحقيقية عصبية، وبدون تسطيح تحصل على البوب.
- **Single-shot error handling.**أنابيب الصوت يجب أن تكون مقاومة للصدمات استثناء واحد يقتل الجلسة

## أرسله

إبقوا`outputs/skill-realtime-designer.md`تصميم خط أذاعة في الوقت الحقيقي مع ميزانيات تأخير ملموسة لكل مرحلة.

## التمارين

1. **Easy.**أركض`code/main.py`يحاكي حلقة حافظة + VAD الطاقة؛ يطبخ التأخيرات المرحلة لتدفق مزيف 10 ثوان.
2. **Medium.**استخدام`sounddevice`،بناء حلقة عبورية التي تعالج الميكروفون في 20ms الإطار وطباعة حالة VAD في كل إطار.
3. **Hard.**قم ببناء اختبار إيقاع مزدوج كامل مع `aiortc`: متصفح → WebRTC → Python → WebRTC → متصفح. قياس تأخير الزجاج إلى الزجاج مع نبض 1 كيلو هرتز.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## المزيد من القراءة

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743)-تقطيعاً من "سيسبر"
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) تأخير كامل من 200 ميس
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) إنتاج وكيل صوتي
- [Silero VAD repo](https://github.com/snakers4/silero-vad) sub-1 ms VAD، Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) إلغاء الصوت تحت المصدر المفتوح.
