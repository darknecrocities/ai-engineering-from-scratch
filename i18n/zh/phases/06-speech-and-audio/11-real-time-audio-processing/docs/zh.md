# 实时音频处理

> 批量管道处理一个文件. 实时管道处理下20毫秒前,下20毫秒到来. 每个对话人工智能,广播工作室和电话机器人都以这个延迟预算生活和死亡.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 问题

您想要一个感觉活跃的语音助理. 人类对话转换延迟约230ms (沉默回应).500ms以上的任何东西都感觉机器人;1500ms以上的东西感觉破碎.**hear → understand → respond → speak**2026年循环是:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

莫希 (九台,2024) 完成了200 ms的全双重时钟.GPT-4o实时 (2024) 钟~320 ms. 2022 年的管道以 2500 ms 运输. 10 倍的改进来自三个技术: (1) 流通到处, (2) 有部分结果的异步管道, (3) 可断断的生成.

## 概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.**现实时间音频流动是固定尺寸的块. 常见选择:20 ms (320 个样本在 16 kHz). 下游的所有东西都必须跟上这个节奏.

**Ring buffer.**固体尺寸圆形缓冲器.生产线写出新的框架,消费线阅读.防止在热路中的分配. 尺寸≈最大延迟 ×样品速率; 2 秒 16 kHz 环 = 32,000 样品.

**VAD (Voice Activity Detection).**通过"Silero VAD 4.0 (2024) "在CPU上运行每30ms的框架. `webrtcvad`现在,我们需要一个更好的选择.

**Streaming ASR.**通过传输方式 (NeMo, 2024) 发射部分转录的模型. 子-CTC-0.6B 在流媒体模式下 (NeMo, 2024) 在 320 ms 延迟时实现25% WER. 声流 (Macháček等, 2023) 块 Whisper 在 ~ 2 秒的延迟时进行近流.

**Interruption.**当用户在助理在说话时,你必须 (a) 检测到入, (b) 停止TTS, (c) 丢弃剩余的LLM输出.所有这些都在100ms内,否则用户会感知助理是聋的.

**WebRTC Opus transport.**浏览器和移动设备的标准. 莱夫基特,日报.co,皮昂是2026年建立语音应用程序的堆.

**Jitter buffer.**网络包裹到达时间已过时. 节缓冲器重新排序和平滑; 太小 → 听力间隙,太大 → 延迟. 典型的6080 ms.

### 常见的

- **Thread contention.**通过使用C-callback音频库 (音频设备,PortAudio) 让Python远离热线.
- **Sample-rate conversion latency.**输入管道内重新样本增加520ms.`soxr_hq`)
- **TTS priming.**即使像Kokoro这样的快速TTS也可以在第一次请求时加热100200ms.缓存模型+在第一次真正转之前用模拟运行加热.
- **Echo cancellation.**没有AEC,TTS输出重新进入麦克风,并触发机器人自己的声音上的ASR.WebRTC AEC3是开源默认的.

```figure
nyquist-aliasing
```

## 建立它

### 步骤1:环保器

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

容量决定了最大缓冲延迟.

### 步骤2:VAD门

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

在生产中用Silero VAD取代:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤3: 流媒体ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤4: 断路处理器

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

网络网络同行连接.停止() 在音频轨道是正规的方式.

## 用它

现在,我们要做什么?

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## 陷

- **Buffering 500 ms to be safe.**缓冲器是你的延迟地板.
- **Not pinning threads.**在优先级低于UI线程上的音频回调 = 负载下出现故障.
- **TTS chunks too small.**微分数为200ms,使声码器的文物听起来.
- **No jitter buffer.**实际的网络是紧张的,没有平滑的你得到了爆发.
- **Single-shot error handling.**音频管道必须是防撞的.

## 运送它

保存如`outputs/skill-realtime-designer.md`设计一个实时音频管道,每个阶段的具体延迟预算.

## 运动

1. **Easy.**跑步`code/main.py`模拟环保器+能量VAD; 打印假的10秒流的阶段延迟.
2. **Medium.**使用`sounddevice`通过一个循环,将你的麦克风处理在20毫米的框架中,
3. **Hard.**通过 构建一个完整的双重回声测试`aiortc`通过1kHz脉冲测量玻璃到玻璃延迟.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## 进一步阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743)           
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) 完全双重200 ms延迟
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/)制作音频代理管弦乐.
- [Silero VAD repo](https://github.com/snakers4/silero-vad)下-1 ms VAD,Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/)在开源下回声取消.
