# कैपस्टोन 11  LLM अवलोकन और समान डैशबोर्ड

> लैंगफ्यूज ओपन-कोर चला गया। एरीज़ फीनिक्स ने 2026 के जेएनएआई सेमकॉन्व मानचित्रण प्रकाशित किए। हेलिकोन और ब्रेनट्रस्ट दोनों ने प्रति उपयोगकर्ता लागत श्रेय में दोगुना किया। Traceloop के OpenLLMetry डी-फेक्टो SDK उपकरण बन गया। उत्पादन आकार निशान के लिए क्लिकहाउस, मेटाडेटा के लिए पोस्टग्रेस, यूआई के लिए Next.js और नमूना निशान पर चलने वाले मूल्यांकन कार्यों (डीपईवल, RAGAS, एलएलएम-जजज) की एक छोटी सेना है। एक स्वयं होस्ट किया गया बनाएं, कम से कम चार एसडीके परिवारों से खाएं, और पांच मिनट से कम समय में इंजेक्शन में गिरावट पकड़ने का प्रदर्शन करें।

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## समस्या

2026 में उत्पादन यातायात चला रहे प्रत्येक एआई टीम मॉडल के साथ एक अवलोकन स्तर बनाए रखती है। लागत निर्धारण। पगडंडी का पता लगाना. बहाव निगरानी. जेलब्रेक संकेत. एसएलओ डैशबोर्ड। पीआईआई लीक अलर्ट। ओपन सोर्स संदर्भ  लैंगफ्यूज, फीनिक्स, ओपनएलएलमेट्रिया  ने इनजेस्ट स्कीम के रूप में ओपनटेलीमेट्री जेएनएआई के अर्थशास्त्रिक सम्मेलनों पर एक साथ काम किया। अब आप एक एसडीके के साथ OpenAI, मानव, गूगल, लैंगचेन, LlamaIndex, और वीएलएलएम उपकरण और जहाज संगत अवधि कर सकते हैं।

आप एक स्व-होस्ट किए गए डैशबोर्ड का निर्माण करेंगे जो कम से कम चार एसडीके परिवारों से निगलता है, नमूने वाले निशानों पर मूल्यांकन कार्यों का एक छोटा सेट चलाता है, बहाव का पता लगाता है, और अलर्ट करता है। माप पट्टीः एक जानबूझकर इंजेक्ट किए गए प्रतिगमन (एक प्रॉम्प्ट जो पीआईआई उत्पन्न करना शुरू करता है) को देखते हुए, डैशबोर्ड इसे पकड़ता है और पांच मिनट से कम समय में अलर्ट चलाता है।

## अवधारणा

Ingest OTLP HTTP है। SDK GenAI-semconv स्पैन उत्पन्न करता हैः `gen_ai.system`,`gen_ai.request.model`,`gen_ai.usage.input_tokens`,`gen_ai.response.id`,`llm.prompts`,`llm.completions`. स्तंभ विश्लेषण के लिए ClickHouse में भूमि को फैलता है; मेटाडेटा (उपयोगकर्ता, सत्र, ऐप) पोस्टग्रेस में भूमि।

ईवल नमूने में शामिल निशानों पर बैच जॉब्स के रूप में चलते हैं। डीपईवल वफादारी, विषाक्तता और उत्तर प्रासंगिकता को स्कोर करता है। RAGAS जब निशान पुनर्प्राप्ति संदर्भ को ले जाता है तो पुनर्प्राप्ति मीट्रिक को स्कोर करता है। कस्टम एलएलएम-न्यायाधीश डोमेन-विशिष्ट जांच (पीआईआई लीक, नीति से बाहर प्रतिक्रिया) करते हैं। ईवल रन मूल निशान से जुड़े मूल्यांकन अवधि के रूप में उसी क्लिकहाउस पर वापस लिखते हैं।

ड्रिफ्ट डिटेक्शन घड़ी समय के साथ एम्बेडिंग-स्पेस वितरण (पीएसआई या केएल अंतर पर शीघ्र एम्बेडिंग) के साथ-साथ मूल्यांकन-स्कोर रुझानों को देखता है। अलर्ट्स प्रोमेथियस अलर्ट मैनेजर और फिर स्लैक / पेजरड्यूटी को फ़ीड करते हैं। यूआई Next.js 15 है।

## वास्तुकला

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## स्टैक

- उपभोगः ओपनटेलीमेट्री एसडीके + जेएनएआई सेमेटिक कन्वेंशन; ओटीएलपी एचटीटीपी परिवहन
- कलेक्टरः ओपनटेलीमेट्री कलेक्टर (लागत नियंत्रण के लिए)
- भंडारणः स्पैन के लिए क्लिकहाउस, मेटाडेटा के लिए पोस्टग्रेस, कच्चे घटना संग्रह के लिए एस 3
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix मूल्यांकन पैक, कस्टम LLM-जजज
- ड्रिफ्टः PSI / KL पर pooled prompt embeddings ( वाक्य-ट्रांसफॉर्मर) साप्ताहिक
- चेतावनीः प्रोमेथियस अलर्ट मैनेजर -> स्लैक / पेजर ड्यूटी
- UI: Next.js 15 App Router + Recharts + सर्वर क्रियाएँ
- SDKs बॉक्स से बाहर समर्थितः OpenAI, मानव, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## इसे बनाओ

1. **Collector config.**ओटीएलपी एचटीटीपी रिसीवर के साथ ओपनटेलीमेट्री कलेक्टर, एक पूंछ-सैम्पलर जो 100% त्रुटिपूर्ण निशान और 10% सफलताओं को रखता है, और ClickHouse और S3 के लिए निर्यातक।

2. **ClickHouse schema.**तालिका `spans`GenAI semconv को दर्शाता स्तंभों के साथः `gen_ai_system`,`gen_ai_request_model`,`input_tokens`,`output_tokens`,`latency_ms`,`prompt_hash`,`trace_id`,`parent_span_id`, और JSON बैग के लिए लंबे उपयोगिता लोड. उपयोगकर्ता_id और app_id द्वारा माध्यमिक सूचकांक जोड़ें.

3. **SDK coverage test.**OpenLLMetry ऑटो-इंस्ट्रूमेंट के साथ प्रत्येक SDK (OpenAI, मानव, Google, LangChain, LlamaIndex, vLLM) का उपयोग करके एक छोटा क्लाइंट ऐप लिखें। सत्यापित करें कि प्रत्येक कैनोनिक GenAI स्पैन उत्पन्न करता है जो ClickHouse में लैंड करता है।

4. **Eval jobs.**एक अनुसूचित कार्य पिछले 15 मिनट के नमूने वाले निशानों को पढ़ता है और डीपईवल निष्ठा, विषाक्तता और उत्तर प्रासंगिकता चलाता है। आउटपुट मूल निशान से जुड़े मूल्यांकन अवधि हैं।

5. **Custom LLM-judge.**एक पीआईपी लीक न्यायाधीशः एक प्रतिक्रिया दी गई है, पीआईपी लीक की संभावना का स्कोर करने के लिए एक गार्ड LLM कॉल करें। उच्च स्कोर प्रतिक्रियाएं एक triage कतार में उतरती हैं।

6. **Drift detection.**साप्ताहिक नौकरी इस सप्ताह के pooled शीघ्र एम्बेड और पिछली 4 सप्ताह के आधार रेखा के बीच PSI गणना करता है। यदि PSI सीमा से ऊपर है, अलर्ट।

7. **Dashboard.**Next.js 15 के साथ पृष्ठः अवलोकन (स्पेन/सेकंड, लागत/उपयोगकर्ता, p95 विलंबता), निशान (खोज + झरना), मूल्यांकन (निष्ठा प्रवृत्ति, विषाक्तता), बहाव (समय के साथ पीएसआई), अलर्ट।

8. **Alerting chain.**प्रोमेथियस निर्यातक मूल्यांकन स्कोर संश्लेषकों और विलंबता प्रतिशत को पढ़ता है; चेतावनी के लिए स्लैक और महत्वपूर्ण उल्लंघन के लिए पेजड्यूटी के लिए अलर्टमैनेजर मार्ग।

9. **Regression probe.**बग इंजेक्ट करेंः मूल्यांकन किया गया चैटबॉट 1% समय में नकली SSNs लीक करना शुरू कर देता है। MTTR मापेंः बग से स्लैक अलर्ट तक तैनात।

## इसका प्रयोग करें

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## इसे भेजें

`outputs/skill-llm-observability.md`LLM आवेदन में, डैशबोर्ड अपने निशान को निगलता है, मूल्यांकन चलाता है, बहाव पर अलर्ट करता है, और Next.js में लागत / उपयोगकर्ता विभाजन को सतह पर प्रकट करता है।

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## व्यायाम

1. Haystack फ्रेमवर्क के लिए कस्टम उपकरण जोड़ें. विश्वासियों के साथ क्लिकहाउस में भूमि के लिए कैनोनिक स्पैन सत्यापित करें `gen_ai.*`गुण।

2. एक ही निशान पर फीनिक्स मूल्यांकनकर्ताओं के लिए डीपईवल स्विच. दो मूल्यांकन इंजन के बीच माप स्कोर बहाव.

3. ड्रीफ डिटेक्टर को तेज करेंः वैश्विक स्तर पर नहीं बल्कि ऐप आईडी के अनुसार पीएसआई की गणना करें। ऐप के अनुसार ड्रीफ ट्रेल दिखाएं।

4. "उपयोगकर्ता प्रभाव" पृष्ठ जोड़ेंः प्रति उपयोगकर्ता लागत और स्पार्कलाइन के साथ प्रति उपयोगकर्ता विफलता दर।

5. एक पूंछ नमूना नीति का निर्माण करें जो विषाक्तता के साथ 100% निशानों को > 0.5 प्लस शेष 10% स्तरीकृत नमूना बनाए रखे।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## आगे पढ़ना

- [Langfuse](https://github.com/langfuse/langfuse) खुला-कोर अवलोकन क्षमता मंच
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) बलशाली बहाव समर्थन के साथ वैकल्पिक संदर्भ
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) ऑटो-इंस्ट्रूमेंटेशन एसडीके परिवार
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) सेवन योजना
- [Helicone](https://www.helicone.ai) वैकल्पिक होस्ट की जा सकती है
- [Braintrust](https://www.braintrust.dev) वैकल्पिक मूल्यांकन-पहला मंच
- [ClickHouse documentation](https://clickhouse.com/docs) स्तंभीय अवधि स्टोर
- [DeepEval](https://github.com/confident-ai/deepeval) मूल्यांकनकर्ता पुस्तकालय
