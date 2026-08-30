# फ़ंक्शन कॉलिंग और टूल उपयोग

> एलएलएम कुछ नहीं कर सकते। वे पाठ उत्पन्न करते हैं। यह पूरी क्षमता है। वे मौसम की जांच नहीं कर सकते, डेटाबेस से क्वेरी नहीं कर सकते, ईमेल नहीं भेज सकते, कोड नहीं चला सकते, या फ़ाइल नहीं पढ़ सकते। हर "AI एजेंट" जिसे आपने कभी देखा है वह एक LLM है जो JSON उत्पन्न करता है जो कहता है कि किस फ़ंक्शन को कॉल करना है -- और फिर आपका कोड वास्तव में इसे कॉल करता है। मॉडल मस्तिष्क है। उपकरण हाथ हैं। कार्य कॉल तंत्रिका तंत्र उन्हें जोड़ता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 03 (Structured Outputs)
**Time:** ~75 minutes
**Related:**चरण 11 · 14 (मॉडल कॉन्टेक्स्ट प्रोटोकॉल)  जब एक टूल को होस्ट के बीच साझा किया जाता है, तो इनलाइन फ़ंक्शन-कॉल से एमसीपी सर्वर तक स्नातक करें। यह पाठ इनलाइन केस को कवर करता है; एमसीपी प्रोटोकॉल केस को कवर करता है।

## सीखने के लक्ष्य

- फ़ंक्शन कॉल लूप को लागू करेंः टूल स्कीम परिभाषित करें, मॉडल के टूल-कॉल JSON का विश्लेषण करें, फ़ंक्शन निष्पादित करें और परिणाम वापस करें
- स्पष्ट विवरण और टाइप किए गए पैरामीटर के साथ डिजाइन उपकरण योजनाएं जिन्हें मॉडल विश्वसनीय रूप से कॉल कर सकता है
- जटिल प्रश्नों के उत्तर देने के लिए कई फ़ंक्शन कॉल को चेन करने वाले एक बहु-टर्न एजेंट लूप बनाएं
- एज केस को कॉल करने वाले हैंडल फ़ंक्शनः समानांतर उपकरण कॉल, त्रुटि प्रसार, और असीमित उपकरण लूप को रोकने

## समस्या

आप एक चैटबॉट बनाते हैं. एक उपयोगकर्ता पूछता है, "टोक्यो में मौसम कैसा है अभी?"

मॉडल जवाब देता हैः "मेरे पास वास्तविक समय मौसम डेटा तक पहुंच नहीं है, लेकिन मौसम के आधार पर, टोक्यो लगभग 15 डिग्री सेल्सियस है . . . "

यह एक भ्रम है जो एक अस्वीकरण के रूप में पहना हुआ है। मॉडल मौसम नहीं जानता है। यह कभी नहीं होगा। मौसम हर घंटे बदलता है। मॉडल के प्रशिक्षण डेटा महीनों पुराना है।

सही उत्तर के लिए OpenWeatherMap API को कॉल करना आवश्यक है, वर्तमान तापमान प्राप्त करना और वास्तविक संख्या लौटा देना। मॉडल एपीआई को नहीं बुला सकता है। आपका कोड कर सकता है। लापता टुकड़ाः एक संरचित प्रोटोकॉल जो मॉडल को यह कहने देता है "मुझे इन तर्कों के साथ मौसम एपीआई को कॉल करने की आवश्यकता है" और आपके कोड को इसे निष्पादित करने देता है और परिणाम वापस फ़ीड करता है।

यह फ़ंक्शन कॉल है. मॉडल संरचनात्मक JSON आउटपुट करता है जो वर्णन करता है कि किस फ़ंक्शन को किस तर्क के साथ बुलाया जाना है। आपका आवेदन फ़ंक्शन को निष्पादित करता है। परिणाम बातचीत में वापस जाता है। मॉडल अपने अंतिम उत्तर को उत्पन्न करने के लिए परिणाम का उपयोग करता है।

कार्य कॉल के बिना, LLM ज्ञानकोश हैं। इसके साथ, वे एजेंट बन जाते हैं।

## अवधारणा

### लूप कॉल करने वाला कार्य

प्रत्येक उपकरण-उपयोग बातचीत एक ही 5-चरण लूप का पालन करती है।

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

चरण 1: उपयोगकर्ता संदेश भेजता है। चरण 2: मॉडल को उपकरण परिभाषाओं के साथ संदेश प्राप्त होता है (जिसॉन योजना उपलब्ध कार्यों का वर्णन करती है) । चरण 3: पाठ के साथ प्रतिक्रिया के बजाय, मॉडल एक उपकरण कॉल आउटपुट करता है - एक संरचित JSON ऑब्जेक्ट फ़ंक्शन नाम और तर्क के साथ। चरण 4: आपका कोड फ़ंक्शन निष्पादित करता है और परिणाम को कैप्चर करता है। चरण 5: परिणाम मॉडल पर वापस जाता है, जिसके पास अब अंतिम उत्तर प्राप्त करने के लिए वास्तविक डेटा है।

मॉडल कभी कुछ भी निष्पादित नहीं करता है यह केवल तय करता है कि क्या कॉल करना है और किस तर्क के साथ। आपका कोड निष्पादक है।

### उपकरण परिभाषाएँः JSON स्कीमा अनुबंध

प्रत्येक उपकरण को JSON स्कीम द्वारा परिभाषित किया जाता है जो मॉडल को बताता है कि फ़ंक्शन क्या करता है, यह किस प्रकार के तर्क लेता है, और उन तर्कों के प्रकार क्या होने चाहिए।

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description`"एक शहर के लिए वर्तमान मौसम प्राप्त करें। तापमान सेल्सियस और परिस्थितियों में लौटाता है।" विवरण उपकरण चयन के लिए एक संकेत है।

### प्रदाता तुलना

प्रत्येक प्रमुख प्रदाता फ़ंक्शन कॉल का समर्थन करता है, लेकिन एपीआई सतह अलग है।

| Provider | API Parameter | Tool Call Format | Parallel Calls | Forced Calling |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | Yes (multiple per turn) | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | Yes (multiple blocks) | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | Yes | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Native `tools` on Llama 4; Hermes or ChatML on others | Mixed | Model-dependent | Prompt-based or `tool_choice` if supported |

2026 तक तीन बंद प्रदाताओं ने लगभग समान JSON-Schema- आधारित प्रारूपों पर एक साथ इकट्ठा हो गए हैं।`tools`ओपनएआई के आकार से मेल खाने वाला फ़ील्ड। ओपन-वेट फाइनट्यून्स अभी भी भिन्न होते हैं  हर्मेस प्रारूप (नोस रिसर्च) तृतीय-पक्ष फाइनट्यून्स के लिए सबसे आम है। होस्ट के बीच साझा किए गए उपकरणों के लिए, इनलाइन फ़ंक्शन-कॉल करने पर MCP (चरण 11 · 14) को प्राथमिकता दें  सर्वर उन सभी के लिए समान है।

### उपकरण विकल्पः ऑटो, आवश्यक, विशिष्ट

आप नियंत्रण जब मॉडल उपकरण का उपयोग करता है.

**Auto**(डिफ़ॉल्ट): मॉडल तय करता है कि क्या एक उपकरण को कॉल करना है या सीधे जवाब देना है. "2 + 2 क्या है?" - सीधे जवाब देता है. "वेदर क्या है?" - उपकरण को बुलाता है.

**Required**उदाहरण: मॉडल को कम से कम एक उपकरण को कॉल करना चाहिए। इसका उपयोग तब करें जब आप जानते हों कि उपयोगकर्ता के इरादे को उपकरण की आवश्यकता होती है। यह मॉडल को वास्तविक डेटा की तलाश करने के बजाय अनुमान लगाने से रोकता है।

**Specific function**: मॉडल को किसी विशेष फ़ंक्शन को कॉल करने के लिए मजबूर करें। `tool_choice={"type":"function", "function": {"name": "get_weather"}}`यह सुनिश्चित करता है कि मौसम उपकरण बुलाया जाता है, चाहे क्वेरी. रूटिंग के लिए इसका उपयोग करें - जब अपस्ट्रीम तर्क पहले से ही निर्धारित किया है कि उपकरण की जरूरत है.

### समानांतर कार्य कॉल

GPT-4o और क्लाउड एक ही बारी में कई कार्यों को कॉल कर सकते हैं। एक उपयोगकर्ता पूछता हैः "टोक्यो और न्यूयॉर्क में मौसम क्या है?" मॉडल एक ही समय में दो उपकरण कॉल आउटपुट करता हैः

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

आपका कोड दोनों को निष्पादित करता है (आदर्श रूप से एक साथ), दोनों परिणाम देता है, और मॉडल एक एकल प्रतिक्रिया संश्लेषित करता है। यह 2 से 1 तक घूमने-फिरने की यात्रा को कम करता है। प्रति क्वेरी 5-10 टूल कॉल वाले एजेंटों के लिए, समानांतर कॉल 60-80% तक विलंबता कम करता है।

### संरचनात्मक आउटपुट बनाम फ़ंक्शन कॉल

पाठ 03 में संरचित आउटपुट शामिल थे। फ़ंक्शन कॉलिंग में एक ही JSON स्कीमा मशीन का उपयोग किया जाता है, लेकिन एक अलग उद्देश्य के लिए।

**Structured outputs**उदाहरण: उत्पाद जानकारी को पाठ से निकालें जैसे कि`{name, price, in_stock}`. .

**Function calling**: मॉडल एक कार्य को निष्पादित करने का इरादा घोषित करता है। आउटपुट एक मध्यवर्ती चरण है। उदाहरण: `get_weather(city="Tokyo")`-- मॉडल एक कार्रवाई का अनुरोध कर रहा है, अंतिम उत्तर नहीं पैदा कर रहा है।

जब आप डेटा निष्कर्षण चाहते हैं तो संरचित आउटपुट का उपयोग करें। जब आप मॉडल को बाहरी प्रणालियों के साथ बातचीत करना चाहते हैं तो फ़ंक्शन कॉल का उपयोग करें।

### सुरक्षाः गैर-विमर्श योग्य नियम

फ़ंक्शन कॉलिंग सबसे खतरनाक क्षमता है जिसे आप एलएलएम दे सकते हैं। मॉडल चुनता है कि क्या निष्पादित करना है। यदि आपके टूल सेट में डेटाबेस क्वेरी शामिल हैं, तो मॉडल क्वेरी का निर्माण करता है। यदि इसमें शेल कमांड शामिल हैं, तो मॉडल उन्हें लिखता है।

**Rule 1: Never pass model-generated SQL directly to a database.**मॉडल ड्रॉप टेबल, यूनियन इंजेक्शन या क्वेरी उत्पन्न कर सकता है और करेगा जो हर पंक्ति को वापस करता है। हमेशा पैरामीटर। हमेशा मान्य करें। हमेशा ऑपरेशन की अनुमति सूची का उपयोग करें।

**Rule 2: Allowlist functions.**मॉडल केवल आपके द्वारा स्पष्ट रूप से परिभाषित कार्यों को कॉल कर सकता है। कभी भी एक सामान्य "नाम द्वारा किसी भी कार्य को निष्पादित करें" उपकरण न बनाएं। यदि आपके पास 50 आंतरिक कार्य हैं, तो केवल 5 को उजागर करें जो उपयोगकर्ता की आवश्यकता है।

**Rule 3: Validate arguments.**मॉडल शहर के नाम से गुजर सकता है `"; DROP TABLE users; --"`. निष्पादन से पहले अपेक्षित प्रकारों, सीमाओं और प्रारूपों के विरुद्ध प्रत्येक तर्क को मान्य करें।

**Rule 4: Sanitize tool results.**यदि कोई उपकरण संवेदनशील डेटा (एपीआई कुंजी, पीआईआई, आंतरिक त्रुटियां) लौटाता है, तो इसे मॉडल पर वापस भेजने से पहले फ़िल्टर करें। मॉडल में इसका जवाब शब्दशः में शामिल किया जाएगा।

**Rule 5: Rate limit tool calls.**एक लूप में एक मॉडल उपकरण को सैकड़ों बार बुला सकता है। अधिकतम सेट करें (10-20 कॉल प्रति बातचीत उचित है) । अंतहीन लूप तोड़ें।

### त्रुटि संभाल

उपकरण विफल होते हैं. एपीआई समय समाप्त होता है. डेटाबेस गिर जाते हैं. फ़ाइलें मौजूद नहीं हैं. मॉडल को पता होना चाहिए कि उपकरण कब विफल होता है और क्यों।

संरचनात्मक उपकरण परिणामों के रूप में त्रुटियों को लौटाएं, अपवाद नहींः

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

मॉडल इसे पढ़ता है, अपने तर्क को समायोजित करता है, और पुनः प्रयास करता है। मॉडल संरचित त्रुटि संदेशों से स्वयं को सुधारने में अच्छे हैं। वे खाली प्रतिक्रियाओं या सामान्य "कुछ गलत चला गया" त्रुटियों से ठीक होने में खराब हैं।

### एमसीपीः मॉडल संदर्भ प्रोटोकॉल

एमसीपी उपकरण सहकार्य के लिए एंथ्रोपिक का खुला मानक है। प्रत्येक एप्लिकेशन के बजाय अपने स्वयं के उपकरणों को परिभाषित करने के लिए, एमसीपी एक सार्वभौमिक प्रोटोकॉल प्रदान करता हैः उपकरण एमसीपी सर्वर द्वारा सेवा प्रदान किए जाते हैं, जो एमसीपी क्लाइंट (जैसे क्लाउड कोड, पाठ्यक्रम या आपके एप्लिकेशन) द्वारा खपत होते हैं।

एक MCP सर्वर किसी भी संगत क्लाइंट के लिए उपकरण उजागर कर सकता है। एक Postgres MCP सर्वर किसी भी MCP संगत एजेंट डेटाबेस एक्सेस देता है। एक GitHub MCP सर्वर किसी भी एजेंट रिपॉजिटरी एक्सेस देता है। उपकरण एक बार परिभाषित किए जाते हैं, हर जगह उपयोग किए जाते हैं।

MCP नेटवर्क को HTTP क्या है, यह कॉल करने के लिए काम करता है। यह परिवहन परत को मानकीकृत करता है ताकि उपकरण पोर्टेबल हो जाएं।

```figure
mx-tool-call-loop
```

## इसे बनाओ

### चरण 1: उपकरण रजिस्ट्री को परिभाषित करें

एक रजिस्ट्री बनाएं जो टूल परिभाषाओं और उनके कार्यान्वयन को संग्रहीत करता है। प्रत्येक टूल में एक JSON स्कीमा परिभाषा (जो मॉडल देखता है) और एक पायथन फ़ंक्शन (जो आपका कोड निष्पादित करता है) होता है।

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### चरण 2: पांच उपकरण लागू करें

एक कैलकुलेटर, मौसम खोज, वेब खोज सिम्युलेटर, फ़ाइल रीडर, और कोड रनर बनाएं।

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### चरण 3: सभी उपकरण पंजीकृत करें

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### चरण 4: कॉल लूप फ़ंक्शन बनाएं

यह मूल इंजन है. यह मॉडल का अनुकरण करता है, यह तय करता है कि किस उपकरण को कॉल करना है, उपकरण को निष्पादित करता है, और परिणाम वापस फ़ीड करता है।

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### चरण 5: तर्क की पुष्टि

एक सत्यापनकर्ता बनाएँ जो निष्पादन से पहले JSON योजना के साथ उपकरण कॉल तर्कों की जांच करता है।

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### चरण 6: डेमो चलाएं

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## इसका प्रयोग करें

### OpenAI फ़ंक्शन कॉल

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI उपकरण कॉल वापस करता है `response.choices[0].message.tool_calls`. प्रत्येक कॉल में एक है `id`आप परिणाम वापस करने के लिए शामिल करना चाहिए. मॉडल कॉल के परिणामों से मेल करने के लिए इस आईडी का उपयोग करता है. GPT-4o एक ही प्रतिक्रिया में कई उपकरण कॉल वापस कर सकते हैं - दोहराएं और उन्हें सभी निष्पादित.

### मानव संसाधन उपकरण का उपयोग

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

एंथ्रोपिक सामग्री ब्लॉक के रूप में उपकरण कॉल वापस करता है `type: "tool_use"`. उपकरण परिणाम के साथ एक उपयोगकर्ता संदेश में जाता है `type: "tool_result"`. मुख्य अंतर ध्यान दें: मानव उपयोग `input_schema`उपकरण पैरामीटर परिभाषाओं के लिए, जबकि OpenAI का उपयोग करता है `parameters`. .

### एमसीपी एकीकरण

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP उपकरण कार्यान्वयन को उपकरण खपत से अलग करता है। पोस्टग्रेस सर्वर SQL जानता है। GitHub सर्वर एपीआई जानता है। आपका एजेंट केवल उपकरण का पता लगाता है और कॉल करता है - प्रत्येक एकीकरण के लिए इसके लिए प्रदाता-विशिष्ट कोड की आवश्यकता नहीं है।

## इसे भेजें

यह सबक हमें फल देता है`outputs/prompt-tool-designer.md`-- उपकरण परिभाषाओं को डिजाइन करने के लिए एक पुनः प्रयोज्य प्रॉम्प्ट टेम्पलेट। आप एक उपकरण क्या करना चाहते हैं का विवरण दें, और यह विवरण, प्रकार और प्रतिबंधों के साथ पूर्ण JSON योजना परिभाषा का उत्पादन करता है।

यह भी उत्पादन करता है `outputs/skill-function-calling-patterns.md`-- उत्पादन में फ़ंक्शन कॉल को लागू करने के लिए एक निर्णय ढांचा, जिसमें उपकरण डिजाइन, त्रुटि प्रबंधन, सुरक्षा और प्रदाता-विशिष्ट पैटर्न शामिल हैं।

## व्यायाम

1. **Add a 6th tool: database query.**मेमोरी मे टेबल के साथ एक सिमुलेटेड एसक्यूएल टूल को लागू करें। टूल एक तालिका नाम और फ़िल्टर शर्तों को स्वीकार करता है (क्रू एसक्यूएल नहीं) । सत्यापित करें कि तालिका नाम एक अनुमतियों सूची में है और फ़िल्टर ऑपरेटरों को सीमित किया गया है `=`,`>`,`<`,`>=`,`<=`. JSON के रूप में मिलान पंक्तियों को लौटाएं.

2. **Implement retry with error feedback.**जब कोई टूल कॉल विफल हो जाता है (उदाहरण के लिए, शहर नहीं मिला है), तो त्रुटि संदेश को मॉडल निर्णय फ़ंक्शन में वापस भेजें और उसे अपने तर्क को सुधारने दें। प्रत्येक कॉल में कितनी पुनः प्रयास किए जाते हैं, उसका ट्रैक करें। प्रत्येक टूल कॉल के लिए अधिकतम 3 पुनः प्रयास सेट करें।

3. **Build a multi-step agent.**कुछ प्रश्नों के लिए चेनिंग टूल कॉल की आवश्यकता होती हैः "कॉन्फिग फ़ाइल पढ़ें और मुझे बताएं कि कौन सा मॉडल कॉन्फ़िगर किया गया है, फिर उस मॉडल की कीमतों के लिए वेब पर खोजें।" एक लूप लागू करें जो तब तक चलता है जब तक मॉडल निर्णय नहीं लेता है कि और अधिक उपकरणों की आवश्यकता नहीं है, संचित परिणामों को प्रत्येक निर्णय चरण में पारित करना। अंतहीन लूप को रोकने के लिए 10 पुनरावृत्ति तक सीमित करें।

4. **Measure tool selection accuracy.**अपेक्षित टूल नामों के साथ 30 परीक्षण क्वेरी बनाएं। अपने निर्णय फ़ंक्शन को सभी 30 पर चलाएं और मापें कि सही टूल का चयन करने का कितना प्रतिशत समय है। उपकरण के बीच कौन से क्वेरी सबसे अधिक भ्रम पैदा करती है।

5. **Implement tool call caching.**यदि 60 सेकंड के भीतर एक ही उपकरण को समान तर्कों के साथ बुलाया जाता है, तो पुनः निष्पादित करने के बजाय कैश किए गए परिणाम को लौटाएं।`(tool_name, frozenset(args.items()))`20 सवालों के साथ बातचीत के दौरान कैश हिट दरों को मापें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Function calling | "Tool use" | The model outputs structured JSON describing a function to invoke with specific arguments -- your code executes it, not the model |
| Tool definition | "Function schema" | A JSON Schema object describing a tool's name, purpose, parameters, and types -- the model reads this to decide when and how to use the tool |
| Tool choice | "Calling mode" | Controls whether the model must call a tool (required), may call a tool (auto), or must call a specific tool (named) |
| Parallel calling | "Multi-tool" | The model outputs multiple tool calls in a single turn, reducing round trips -- GPT-4o and Claude both support this |
| Tool result | "Function output" | The return value from executing a tool, sent back to the model as a message so it can use real data in its response |
| Argument validation | "Input checking" | Verifying that model-generated arguments match the expected types, ranges, and constraints before executing the tool |
| MCP | "Tool protocol" | Model Context Protocol -- Anthropic's open standard for exposing tools via servers that any compatible client can discover and call |
| Agent loop | "ReAct loop" | The iterative cycle of model-decides-tool, code-executes-tool, result-feeds-back until the model has enough information to respond |
| Tool poisoning | "Prompt injection via tools" | An attack where tool results contain instructions that manipulate the model's behavior -- sanitize all tool outputs |
| Rate limiting | "Call budget" | Setting a maximum number of tool calls per conversation to prevent infinite loops and runaway API costs |

## आगे पढ़ना

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)-- जीपीटी-4o के साथ उपकरण के उपयोग के लिए अंतिम संदर्भ, जिसमें समानांतर कॉल, मजबूर कॉल और संरचित तर्क शामिल हैं
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)-- क्लाउड के उपकरण उपयोग कार्यान्वयन के साथ input_schema, बहु-उपकरण प्रतिक्रियाओं, और उपकरण_चयन विन्यास
- [Model Context Protocol Specification](https://modelcontextprotocol.io)-- सर्वर/क्लाइंट आर्किटेक्चर के साथ एआई अनुप्रयोगों के बीच उपकरण सहकार्य के लिए खुला मानक
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761)-- बाहरी उपकरणों को कब और कैसे बुलाया जाए, इसके लिए एलएलएम को प्रशिक्षण देने पर आधारभूत पेपर
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334)-- हलुसिनाशन कम करने के साथ 1,645 एपीआई पर सटीक एपीआई कॉल के लिए LLM को ठीक से समायोजित करना
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)-- वास्तविक समय बेंचमार्क की तुलना GPT-4o, क्लाउड, मिथुन, और खुले मॉडल में सटीकता कॉल फ़ंक्शन
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)-- विचार-क्रिया-निरीक्षण लूप जो प्रत्येक उपकरण कॉल के आसपास बाहरी एजेंट लूप है; जहां यह सबक समाप्त होता है, चरण 14 शुरू होता है।
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)-- पांच संरचनात्मक पैटर्न (प्रोम्प्ट चेनिंग, रूटिंग, समानांतर, ऑर्केस्ट्रेटर-वर्कर, मूल्यांकनकर्ता-अनुकूलनकर्ता) एकल उपकरण उपयोग आदिम से बनाया गया।
