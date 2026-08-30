# संचार प्रोटोकॉल

> एक ही भाषा नहीं बोलने वाले एजेंट एक टीम नहीं हैं, वे एक अजनबी हैं जो खाली जगह में चिल्ला रहे हैं।

**Type:** Build
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering), Lesson 16.01 (Why Multi-Agent)
**Time:** ~120 minutes

## सीखने के लक्ष्य

- एमसीपी उपकरण खोज और आवंटन को लागू करें ताकि एजेंट बाहरी सर्वर द्वारा उजागर किए गए उपकरणों का उपयोग कर सकें
- एक ए 2 ए एजेंट कार्ड और कार्य अंत बिंदु बनाएं जो एक एजेंट को HTTP के माध्यम से काम को दूसरे पर सौंपने की अनुमति देता है
- MCP (उपकरण पहुंच), A2A (एजेंट-टू-एजेंट), ACP (उद्यमी लेखा परीक्षा) और ANP (विकेन्द्रीकृत विश्वास) की तुलना करें और बताएं कि कौन सा प्रोटोकॉल किस समस्या को हल करता है
- एक प्रणाली में कई प्रोटोकॉल को एक साथ वायर करें जहां एजेंट MCP के माध्यम से उपकरण खोजते हैं और A2A के माध्यम से कार्य सौंपते हैं

## समस्या

आप अपने सिस्टम को कई एजेंटों में विभाजित करते हैं एक शोधकर्ता, एक कोडर, एक समीक्षक वे अपने व्यक्तिगत काम में महान हैं लेकिन अब आपको उनकी जरूरत है कि वे वास्तव में एक दूसरे से बात करें।

आपका पहला प्रयास स्पष्ट हैः स्ट्रिंग्स को पास करें। शोधकर्ता पाठ का एक ब्लाब लौटाता है, कोडर इसे वैसे भी पार्स करता है जैसे वह कर सकता है। यह तब तक काम करता है जब तक कोडर एक शोध सारांश को गलत तरीके से नहीं समझता है, या दो एजेंट एक दूसरे के लिए इंतजार कर रहे हैं, या आपको सहयोग करने के लिए विभिन्न टीमों द्वारा बनाए गए एजेंटों की आवश्यकता होती है। अचानक "बस स्ट्रिंग्स पास करें" टूट जाता है।

यह संचार प्रोटोकॉल की समस्या है. एजेंटों के साथ सूचना का आदान-प्रदान करने के लिए साझा अनुबंध के बिना, मल्टी-एजेंट सिस्टम नाजुक, अडिट करने योग्य और आपके द्वारा व्यक्तिगत रूप से लिखे गए कुछ एजेंटों से परे पैमाने पर नहीं हो सकते हैं।

एआई पारिस्थितिकी तंत्र ने चार प्रोटोकॉल के साथ प्रतिक्रिया दी है, प्रत्येक समस्या का एक अलग स्लाइस हल करता हैः

- **MCP**उपकरण तक पहुँच के लिए
- **A2A**एजेंट-एजेंट सहयोग के लिए
- **ACP**उद्यम लेखा परीक्षा के लिए
- **ANP**विकेन्द्रीकृत पहचान और विश्वास के लिए

यह सबक गहराई से जाता है. आप प्रत्येक विनिर्देश से वास्तविक तार प्रारूपों को पढ़ेंगे, काम करने वाले कार्यान्वयन का निर्माण करेंगे, और सभी चार को एक एकीकृत प्रणाली में जोड़ेंगे।

## अवधारणा

### प्रोटोकॉल परिदृश्य

इन चार प्रोटोकॉल को एक स्तर के रूप में सोचें, प्रत्येक एक अलग प्रश्न को संबोधित करता हैः

```mermaid
flowchart TD
  ANP["ANP — How do agents trust strangers?<br/>Decentralized identity (DID), E2EE, meta-protocol"]
  A2A["A2A — How do agents collaborate on goals?<br/>Agent Cards, task lifecycle, streaming, negotiation"]
  ACP["ACP — How do agents talk in auditable systems?<br/>Runs, trajectory metadata, session continuity"]
  MCP["MCP — How does an agent use a tool?<br/>Tool discovery, execution, context sharing"]

  style ANP fill:#f3e8ff,stroke:#7c3aed
  style A2A fill:#dbeafe,stroke:#2563eb
  style ACP fill:#fef3c7,stroke:#d97706
  style MCP fill:#d1fae5,stroke:#059669
```

वे प्रतियोगी नहीं हैं, वे अलग-अलग स्तरों पर अलग-अलग समस्याएं हल करते हैं।

### एमसीपी (पुनर्बिन्यास)

एमसीपी चरण 13 में गहराई से कवर किया गया है। त्वरित पुनरावृत्तिः एमसीपी मानकीकृत करता है कि एलएलएम बाहरी उपकरणों और डेटा स्रोतों से कैसे जुड़ता है। यह एक **client-server**प्रोटोकॉल जहां एजेंट (क्लाइंट) सर्वर द्वारा उजागर किए गए उपकरणों की खोज और कॉल करता है।

```mermaid
sequenceDiagram
    participant Agent as Agent (client)
    participant MCP1 as MCP Server<br/>(database, API, files)

    Agent->>MCP1: list tools
    MCP1-->>Agent: tool definitions
    Agent->>MCP1: call tool X
    MCP1-->>Agent: result
```

एमसीपी **agent-to-tool**यह एजेंटों को एक दूसरे के साथ बात करने में मदद नहीं करता है।

### ए2ए (एजेंट2एजेंट प्रोटोकॉल)

**Created by:**गूगल (अब लिनक्स फाउंडेशन के तहत `lf.a2a.v1`)
**Spec version:**1.0.0
**Problem:**स्वायत्त एजेंट कैसे सहयोग करते हैं, बातचीत करते हैं और एक दूसरे को कार्य सौंपते हैं?

A2A के लिए प्रोटोकॉल है**peer-to-peer agent collaboration**. जहां एमसीपी एक एजेंट को उपकरण से जोड़ता है, ए2ए एक एजेंट को अन्य एजेंटों से जोड़ता है। प्रत्येक एजेंट एक **Agent Card**एक प्रसिद्ध यूआरएल पर, और अन्य एजेंटों को पता लगाने, बातचीत करने और उसे कार्य सौंपने के लिए।

#### ए 2 ए कैसे काम करता है

```mermaid
sequenceDiagram
    participant Client as Client Agent
    participant Remote as Remote Agent

    Client->>Remote: GET /.well-known/agent-card.json
    Remote-->>Client: Agent Card (skills, modes, security)

    Client->>Remote: POST /message:send
    Remote-->>Client: Task (submitted/working)

    alt Polling
        Client->>Remote: GET /tasks/{id}
        Remote-->>Client: Task status + artifacts
    else Streaming
        Client->>Remote: POST /message:stream
        Remote-->>Client: SSE: statusUpdate
        Remote-->>Client: SSE: artifactUpdate
        Remote-->>Client: SSE: completed
    end
```

#### असली एजेंट कार्ड

यह है कि एक ए 2 ए एजेंट कार्ड वास्तव में जंगली में कैसा दिखता है.`GET /.well-known/agent-card.json`:

```json
{
  "name": "Research Agent",
  "description": "Searches documentation and summarizes findings",
  "version": "1.0.0",
  "supportedInterfaces": [
    {
      "url": "https://research-agent.example.com/a2a/v1",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    },
    {
      "url": "https://research-agent.example.com/a2a/rest",
      "protocolBinding": "HTTP+JSON",
      "protocolVersion": "1.0"
    }
  ],
  "provider": {
    "organization": "Your Company",
    "url": "https://example.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "skills": [
    {
      "id": "web-research",
      "name": "Web Research",
      "description": "Searches the web and synthesizes findings",
      "tags": ["research", "search", "summarization"],
      "examples": ["Research the latest changes in React 19"]
    },
    {
      "id": "doc-analysis",
      "name": "Documentation Analysis",
      "description": "Reads and analyzes technical documentation",
      "tags": ["docs", "analysis"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {
    "bearer": {
      "httpAuthSecurityScheme": {
        "scheme": "Bearer",
        "bearerFormat": "JWT"
      }
    }
  },
  "security": [{ "bearer": [] }]
}
```

ध्यान देने योग्य महत्वपूर्ण बातेंः
- **Skills**एक ग्राहक एजेंट इस तरह से तय करता है कि क्या यह रिमोट एजेंट अपने अनुरोध को संभाल सकता है।
- **supportedInterfaces**एक एकल एजेंट JSON-RPC, REST, और gRPC एक साथ बोल सकता है।
- **Security**ग्राहक को एक भी अनुरोध करने से पहले ही पता है कि उसे किस लेखक की आवश्यकता है।

#### कार्य जीवन चक्र

कार्य ए 2 ए में काम की मूल इकाई हैं। वे परिभाषित राज्यों के माध्यम से चलते हैंः

```mermaid
stateDiagram-v2
    [*] --> submitted
    submitted --> working
    working --> input_required: needs more info
    input_required --> working: client sends data
    working --> completed: success
    working --> failed: error
    working --> canceled: client cancels
    submitted --> rejected: agent declines

    completed --> [*]
    failed --> [*]
    canceled --> [*]
    rejected --> [*]

    note right of completed
        Terminal states are immutable.
        Follow-ups create new tasks
        within the same contextId.
    end note
```

सभी 8 राज्यों (विशिष्टता भी परिभाषित करती है `UNSPECIFIED`एक प्रहरी के रूप में, यहाँ छोड़ा गया):

| State | Terminal? | Meaning |
|---|---|---|
| `TASK_STATE_SUBMITTED` | No | Acknowledged, not yet processing |
| `TASK_STATE_WORKING` | No | Actively being processed |
| `TASK_STATE_INPUT_REQUIRED` | No | Agent needs more info from client |
| `TASK_STATE_AUTH_REQUIRED` | No | Authentication needed |
| `TASK_STATE_COMPLETED` | Yes | Finished successfully |
| `TASK_STATE_FAILED` | Yes | Finished with error |
| `TASK_STATE_CANCELED` | Yes | Canceled before completion |
| `TASK_STATE_REJECTED` | Yes | Agent declined the task |

एक बार जब कोई कार्य एक टर्मिनल स्थिति तक पहुंच जाता है, तो यह अपरिवर्तनीय होता है। कोई और संदेश नहीं। अनुवर्ती उसी के भीतर एक नया कार्य बनाता है।`contextId`. .

#### तार प्रारूप

A2A JSON-RPC 2.0 का उपयोग करता है। यहाँ एक असली संदेश विनिमय कैसा दिखता हैः

**Client sends a task:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{ "text": "Research React 19 compiler features" }]
    },
    "configuration": {
      "acceptedOutputModes": ["text/plain", "application/json"],
      "historyLength": 10
    }
  }
}
```

**Agent responds with a task:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "task": {
      "id": "task-abc-123",
      "contextId": "ctx-xyz-789",
      "status": {
        "state": "TASK_STATE_COMPLETED",
        "timestamp": "2026-03-27T10:30:00Z"
      },
      "artifacts": [
        {
          "artifactId": "art-001",
          "name": "research-results",
          "parts": [{
            "data": {
              "findings": [
                "React 19 compiler auto-memoizes components",
                "No more manual useMemo/useCallback needed",
                "Compiler runs at build time, not runtime"
              ]
            },
            "mediaType": "application/json"
          }]
        }
      ]
    }
  }
}
```

**Streaming via SSE:**
```text
POST /message:stream HTTP/1.1
Content-Type: application/json
A2A-Version: 1.0

data: {"task":{"id":"task-123","status":{"state":"TASK_STATE_WORKING"}}}

data: {"statusUpdate":{"taskId":"task-123","status":{"state":"TASK_STATE_WORKING","message":{"role":"ROLE_AGENT","parts":[{"text":"Searching documentation..."}]}}}}

data: {"artifactUpdate":{"taskId":"task-123","artifact":{"artifactId":"art-1","parts":[{"text":"partial findings..."}]},"append":true,"lastChunk":false}}

data: {"statusUpdate":{"taskId":"task-123","status":{"state":"TASK_STATE_COMPLETED"}}}
```

### एसीपी (एजेंट संचार प्रोटोकॉल)

**Created by:**आईबीएम / बीआईएआई
**Spec version:**0.2.0 (OpenAPI 3.1.1)
**Status:**लिनक्स फाउंडेशन के तहत A2A में विलय
**Problem:**एजेंट पूर्ण लेखा परीक्षा, सत्र निरंतरता और ट्रैकिंग ट्रैक के साथ कैसे संवाद करते हैं?

ACP **enterprise protocol**. कई सारांशों के विपरीत, एसीपी करता है **not**यह एक सरल REST / JSON एपीआई है OpenAPI के माध्यम से परिभाषित. जो इसे विशेष बनाता है यह है**TrajectoryMetadata**: प्रत्येक एजेंट प्रतिक्रिया तर्क चरणों और उपकरण कॉल जो इसे उत्पन्न की एक विस्तृत लॉग ले जा सकता है।

```mermaid
sequenceDiagram
    participant Client
    participant ACP as ACP Agent
    participant Audit as Audit Log

    Client->>ACP: POST /runs (mode: sync)
    ACP->>ACP: Process request...
    ACP->>Audit: Log trajectory:<br/>reasoning + tool calls
    ACP-->>Client: Response + TrajectoryMetadata
    Note over Audit: Every step recorded:<br/>tool_name, tool_input,<br/>tool_output, reasoning
```

#### एसीपी में एजेंट डिस्कवरी

ACP चार खोज पद्धतियों को परिभाषित करता हैः

```mermaid
graph LR
    A[Agent Discovery] --> B["Runtime<br/>GET /agents"]
    A --> C["Open<br/>.well-known/agent.yml"]
    A --> D["Registry<br/>Centralized catalog"]
    A --> E["Embedded<br/>Container labels"]

    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#d1fae5,stroke:#059669
    style D fill:#fef3c7,stroke:#d97706
    style E fill:#f3e8ff,stroke:#7c3aed
```

**AgentManifest**ए 2 ए के एजेंट कार्ड से सरल हैः

```json
{
  "name": "summarizer",
  "description": "Summarizes documents with source citations",
  "input_content_types": ["text/plain", "application/pdf"],
  "output_content_types": ["text/plain", "application/json"],
  "metadata": {
    "tags": ["summarization", "RAG"],
    "framework": "BeeAI",
    "capabilities": [
      {
        "name": "Document Summarization",
        "description": "Condenses long documents into key points"
      }
    ],
    "recommended_models": ["llama3.3:70b-instruct-fp16"],
    "license": "Apache-2.0",
    "programming_language": "Python"
  }
}
```

#### जीवन चक्र चलाएँ

एसीपी "कार्य" के बजाय "रन्स" का उपयोग करता है। एक रन तीन मोड के साथ एक एजेंट निष्पादन हैः

| Mode | Behavior |
|---|---|
| `sync` | Blocking. Response contains the complete result. |
| `async` | Returns 202 immediately. Poll `GET /runs/{id}` for status. |
| `stream` | SSE stream. Events fire as the agent works. |

```mermaid
stateDiagram-v2
    [*] --> created
    created --> in_progress
    in_progress --> completed: success
    in_progress --> failed: error
    in_progress --> awaiting: needs input
    awaiting --> in_progress: client resumes
    in_progress --> cancelling: cancel request
    cancelling --> cancelled

    completed --> [*]
    failed --> [*]
    cancelled --> [*]
```

#### ट्रैकटोरियममेटाडेटा (ऑडिट ट्रेल)

यह एसीपी का मुख्य भेद है। प्रत्येक संदेश भाग में मेटाडेटा हो सकता है जो एजेंट ने क्या किया है, इसका सटीक रूप से पता चलता हैः

```json
{
  "role": "agent/researcher",
  "parts": [
    {
      "content_type": "text/plain",
      "content": "The weather in San Francisco is 72F and sunny.",
      "metadata": {
        "kind": "trajectory",
        "message": "I need to check the weather for this location",
        "tool_name": "weather_api",
        "tool_input": { "location": "San Francisco, CA" },
        "tool_output": { "temperature": 72, "condition": "sunny" }
      }
    }
  ]
}
```

नियामक उद्योगों के लिए यह सोना है। प्रत्येक उत्तर तर्क की एक सिद्ध श्रृंखला के साथ आता हैः किस उपकरण को बुलाया गया था, किस इनपुट का उपयोग किया गया था, किस आउटपुट को प्राप्त किया गया था। कोई ब्लैक बॉक्स नहीं।

ACP भी समर्थन करता है **CitationMetadata**स्रोत श्रेणियों के लिएः

```json
{
  "kind": "citation",
  "start_index": 0,
  "end_index": 47,
  "url": "https://weather.gov/sf",
  "title": "NWS San Francisco Forecast"
}
```

### एएनपी (एजेंट नेटवर्क प्रोटोकॉल)

**Created by:**ओपन सोर्स समुदाय (गौवेई चांग द्वारा स्थापित)
**Repo:** [github.com/agent-network-protocol/AgentNetworkProtocol](https://github.com/agent-network-protocol/AgentNetworkProtocol)
**Problem:**विभिन्न संगठनों के एजेंट बिना किसी केंद्रीय प्राधिकरण के एक दूसरे पर कैसे भरोसा करते हैं?

एएनपी **decentralized identity protocol**. यह W3C विकेंद्रीकृत पहचानकर्ताओं (DIDs) और अंत-से-अंत एन्क्रिप्शन का उपयोग करके विश्वास का निर्माण करता है। ए 2 ए के विपरीत जहां आप ज्ञात एंडपॉइंट के माध्यम से एजेंटों की खोज करते हैं, एएनपी एजेंटों को अपनी पहचान को क्रिप्टोग्राफिक रूप से साबित करने देता है।

एएनपी में तीन परतें हैंः

```mermaid
graph TB
    subgraph Layer3["Layer 3: Application Protocol"]
        AD[Agent Description Documents]
        DISC[Discovery endpoints]
    end
    subgraph Layer2["Layer 2: Meta-Protocol"]
        NEG[AI-powered protocol negotiation]
        CODE[Dynamic code generation]
    end
    subgraph Layer1["Layer 1: Identity & Secure Communication"]
        DID["did:wba (W3C DID)"]
        HPKE[HPKE E2EE - RFC 9180]
        SIG[Signature verification]
    end

    Layer3 --> Layer2
    Layer2 --> Layer1

    style Layer1 fill:#d1fae5,stroke:#059669
    style Layer2 fill:#dbeafe,stroke:#2563eb
    style Layer3 fill:#f3e8ff,stroke:#7c3aed
```

#### डीआईडी दस्तावेज (वास्तविक संरचना)

एएनपी एक कस्टम डीआईडी विधि का उपयोग करता है जिसे कहा जाता है `did:wba`(वेब आधारित एजेंट)`did:wba:example.com:user:alice`निर्णय लेता है `https://example.com/user/alice/did.json`:

```json
{
  "@context": [
    "https://www.w3.org/ns/did/v1",
    "https://w3id.org/security/suites/jws-2020/v1",
    "https://w3id.org/security/suites/secp256k1-2019/v1"
  ],
  "id": "did:wba:example.com:user:alice",
  "verificationMethod": [
    {
      "id": "did:wba:example.com:user:alice#key-1",
      "type": "EcdsaSecp256k1VerificationKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyJwk": {
        "crv": "secp256k1",
        "x": "NtngWpJUr-rlNNbs0u-Aa8e16OwSJu6UiFf0Rdo1oJ4",
        "y": "qN1jKupJlFsPFc1UkWinqljv4YE0mq_Ickwnjgasvmo",
        "kty": "EC"
      }
    },
    {
      "id": "did:wba:example.com:user:alice#key-x25519-1",
      "type": "X25519KeyAgreementKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyMultibase": "z9hFgmPVfmBZwRvFEyniQDBkz9LmV7gDEqytWyGZLmDXE"
    }
  ],
  "authentication": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "keyAgreement": [
    "did:wba:example.com:user:alice#key-x25519-1"
  ],
  "humanAuthorization": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "service": [
    {
      "id": "did:wba:example.com:user:alice#agent-description",
      "type": "AgentDescription",
      "serviceEndpoint": "https://example.com/agents/alice/ad.json"
    }
  ]
}
```

ध्यान देने योग्य महत्वपूर्ण बातेंः
- **Key separation**हस्ताक्षर कुंजी (secp256k1) एन्क्रिप्शन कुंजी (X25519) से अलग है।
- **`humanAuthorization`**इन कुंजी को उपयोग करने से पहले स्पष्ट मानव अनुमोदन (बायोमेट्रिक, पासवर्ड, एचएसएम) की आवश्यकता होती है। धन हस्तांतरण जैसे उच्च जोखिम वाले संचालन इस मार्ग से गुजरते हैं।
- **`keyAgreement`**HPKE अंत-से-अंत एन्क्रिप्शन (RFC 9180) के लिए कुंजी का उपयोग किया जाता है।
- **service**एजेंट विवरण दस्तावेज के लिए अनुभाग लिंक।

#### एएनपी में विश्वास कैसे काम करता है

एएनपी करता है **not**विश्वास के वेब या अनुमोदन ग्राफ का उपयोग करें। विश्वास द्विपक्षीय है और प्रति बातचीत सत्यापित किया जाता हैः

```mermaid
sequenceDiagram
    participant A as Agent A
    participant Domain as Agent A's Domain
    participant B as Agent B

    A->>B: HTTP request + DID + signature
    B->>Domain: Fetch DID document (HTTPS)
    Domain-->>B: DID document + public key
    B->>B: Verify signature with public key
    B-->>A: Issue access token
    A->>B: Subsequent requests use token
    Note over A,B: Trust = TLS domain verification<br/>+ DID signature verification<br/>+ Principle of least trust
```

विश्वास तीन स्रोतों से आता हैः
1. **Domain-level TLS**डीआईडी दस्तावेज़ होस्ट की पुष्टि करता है
2. **DID cryptographic signatures**एजेंट की पहचान सत्यापित करें
3. **Principle of least trust**केवल न्यूनतम अनुमति देता है

कोई गपशप आधारित विश्वास प्रचार या पेज रैंक स्कोर नहीं है. आप प्रत्येक एजेंट को सीधे अपने डीआईडी के माध्यम से सत्यापित करते हैं.

#### मेटा प्रोटोकॉल वार्ता

यह एएनपी की सबसे नवीन विशेषता है जब दो एजेंट अलग-अलग पारिस्थितिकी तंत्र से मिलते हैं, उन्हें पूर्व-समझी हुई डेटा प्रारूपों की आवश्यकता नहीं होती है। वे प्राकृतिक भाषा में बातचीत करते हैंः

```json
{
  "action": "protocolNegotiation",
  "sequenceId": 0,
  "candidateProtocols": "I can communicate using:\n1. JSON-RPC with hotel booking schema\n2. REST with OpenAPI 3.1 spec\n3. Natural language over HTTP",
  "modificationSummary": "Initial proposal",
  "status": "negotiating"
}
```

```mermaid
sequenceDiagram
    participant A as Agent A
    participant B as Agent B

    A->>B: protocolNegotiation (candidateProtocols)
    B->>A: protocolNegotiation (counter-proposal)
    A->>B: protocolNegotiation (accepted)
    Note over A,B: Agents dynamically generate code<br/>to handle the agreed format.<br/>Max 10 rounds, then timeout.
```

एजेंट एक प्रारूप पर सहमत होने तक आगे और पीछे (मैक्स 10 राउंड) जाते हैं, फिर इसे संभालने के लिए गतिशील रूप से कोड उत्पन्न करते हैं। स्थिति मानः `negotiating`,`rejected`,`accepted`,`timeout`. .

इसका मतलब है कि दो एजेंट जो पहले कभी एक दूसरे को नहीं देखे हैं, बिना किसी साझा योजना को पूर्व परिभाषित किए संवाद करने का तरीका समझ सकते हैं।

### तुलना (संदिग्ध)

| | MCP | A2A | ACP | ANP |
|---|---|---|---|---|
| **Created by** | Anthropic | Google / Linux Foundation | IBM / BeeAI | Community |
| **Spec format** | JSON-RPC | JSON-RPC / REST / gRPC | OpenAPI 3.1 (REST) | JSON-RPC |
| **Primary use** | Agent to Tool | Agent to Agent | Agent to Agent | Agent to Agent |
| **Discovery** | Tool listing | `/.well-known/agent-card.json` | `GET /agents`, `/.well-known/agent.yml` | `/.well-known/agent-descriptions`, DID service endpoints |
| **Identity** | Implicit (local) | Security schemes (OAuth, mTLS) | Server-level | W3C DID (`did:wba`) with E2EE |
| **Audit trail** | N/A | Basic (task history) | TrajectoryMetadata (tool calls, reasoning) | Not formally specified |
| **State machine** | N/A | 9 task states | 7 run states | N/A |
| **Streaming** | N/A | SSE | SSE | Transport-agnostic |
| **Unique feature** | Tool schemas | Agent Cards + Skills | Trajectory audit trail | Meta-protocol negotiation |
| **Best for** | Tools & data | Dynamic collaboration | Regulated industries | Cross-org trust |
| **Status** | Stable | Stable (v1.0) | Merging into A2A | Active development |

### कैसे वे एक साथ काम करते हैं

एक यथार्थवादी उद्यम प्रणाली में कई प्रकार के प्रोटोकॉल होते हैंः

```mermaid
graph TB
    subgraph org["Your Organization"]
        RA[Research Agent] <-->|A2A| CA[Coding Agent]
        RA -->|MCP| SS[Search Server]
        CA -->|MCP| GS[GitHub Server]
        AUDIT["All agent responses carry<br/>ACP TrajectoryMetadata"]
    end

    subgraph ext["External (DID verified via ANP)"]
        EA[External Agent]
        PA[Partner Agent]
    end

    RA <-->|ANP + A2A| EA
    CA <-->|ANP + A2A| PA

    style org fill:#f8fafc,stroke:#334155
    style ext fill:#fef2f2,stroke:#991b1b
    style AUDIT fill:#fef3c7,stroke:#d97706
```

- **MCP**प्रत्येक एजेंट को उसके उपकरण से जोड़ता है
- **A2A**एजेंटों के बीच सहयोग (आंतरिक और बाहरी) को संभालता है
- **ACP**लेखापरीक्षा के लिए ट्रैकटोरिया मेटाडेटा में प्रतिक्रियाओं को लपेटता है
- **ANP**आप नियंत्रण नहीं कर रहे एजेंटों के लिए पहचान सत्यापन प्रदान करता है

```figure
swarm-message-bus
```

## इसे बनाओ

### चरण 1: मुख्य संदेश प्रकार

प्रत्येक बहु एजेंट प्रणाली संदेश प्रारूप के साथ शुरू होता है. हम प्रकारों को परिभाषित करते हैं जो वास्तविक प्रोटोकॉल का उपयोग करते हैं के लिए मैप करते हैंः

```typescript
import crypto from "node:crypto";

type MessageRole = "user" | "agent";

type MessagePart =
  | { kind: "text"; text: string }
  | { kind: "data"; data: unknown; mediaType: string }
  | { kind: "file"; name: string; url: string; mediaType: string };

type TrajectoryEntry = {
  reasoning: string;
  toolName?: string;
  toolInput?: unknown;
  toolOutput?: unknown;
  timestamp: number;
};

type AgentMessage = {
  id: string;
  role: MessageRole;
  parts: MessagePart[];
  trajectory?: TrajectoryEntry[];
  replyTo?: string;
  timestamp: number;
};

function createMessage(
  role: MessageRole,
  parts: MessagePart[],
  replyTo?: string
): AgentMessage {
  return {
    id: crypto.randomUUID(),
    role,
    parts,
    replyTo,
    timestamp: Date.now(),
  };
}

function textMessage(role: MessageRole, text: string): AgentMessage {
  return createMessage(role, [{ kind: "text", text }]);
}
```

ध्यान दें: `MessagePart`यह मल्टीमोडल (टेक्स्ट, संरचित डेटा, फाइलें) है, जैसे वास्तविक A2A और ACP विनिर्देश। `TrajectoryEntry`इस प्रकार, एसीपी के ट्रैकटोरियामेटाडेटा से मेल खाने वाली तर्क श्रृंखला को कैप्चर करता है।

### चरण 2: ए2ए एजेंट कार्ड और रजिस्ट्री

वास्तविक A2A विनिर्देशों से मेल खाने वाले एजेंट खोज का निर्माण करेंः

```typescript
type Skill = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  inputModes: string[];
  outputModes: string[];
};

type AgentCard = {
  name: string;
  description: string;
  version: string;
  url: string;
  capabilities: {
    streaming: boolean;
    pushNotifications: boolean;
  };
  defaultInputModes: string[];
  defaultOutputModes: string[];
  skills: Skill[];
};

class AgentRegistry {
  private cards: Map<string, AgentCard> = new Map();

  register(card: AgentCard) {
    this.cards.set(card.name, card);
  }

  discoverBySkillTag(tag: string): AgentCard[] {
    return [...this.cards.values()].filter((card) =>
      card.skills.some((skill) => skill.tags.includes(tag))
    );
  }

  discoverByInputMode(mimeType: string): AgentCard[] {
    return [...this.cards.values()].filter(
      (card) =>
        card.defaultInputModes.includes(mimeType) ||
        card.skills.some((skill) => skill.inputModes.includes(mimeType))
    );
  }

  resolve(name: string): AgentCard | undefined {
    return this.cards.get(name);
  }

  listAll(): AgentCard[] {
    return [...this.cards.values()];
  }
}
```

यह एक साधारण नाम-से-क्षमता मानचित्र से काफी अधिक समृद्ध है. आप कौशल टैग द्वारा एजेंटों की खोज कर सकते हैं, इनपुट MIME प्रकारों द्वारा, या नाम से, बस वास्तविक A2A विनिर्देश समर्थन की तरह.

### चरण 3: ए 2 ए कार्य जीवन चक्र

पूर्ण कार्य राज्य मशीन का निर्माण करेंः

```typescript
type TaskState =
  | "submitted"
  | "working"
  | "input-required"
  | "auth-required"
  | "completed"
  | "failed"
  | "canceled"
  | "rejected";

const TERMINAL_STATES: TaskState[] = [
  "completed",
  "failed",
  "canceled",
  "rejected",
];

type TaskStatus = {
  state: TaskState;
  message?: AgentMessage;
  timestamp: number;
};

type Artifact = {
  id: string;
  name: string;
  parts: MessagePart[];
};

type Task = {
  id: string;
  contextId: string;
  status: TaskStatus;
  artifacts: Artifact[];
  history: AgentMessage[];
};

type TaskEvent =
  | { kind: "statusUpdate"; taskId: string; status: TaskStatus }
  | {
      kind: "artifactUpdate";
      taskId: string;
      artifact: Artifact;
      append: boolean;
      lastChunk: boolean;
    };

type TaskHandler = (
  task: Task,
  message: AgentMessage
) => AsyncGenerator<TaskEvent>;

class TaskManager {
  private tasks: Map<string, Task> = new Map();
  private handlers: Map<string, TaskHandler> = new Map();
  private listeners: Map<string, ((event: TaskEvent) => void)[]> = new Map();

  registerHandler(agentName: string, handler: TaskHandler) {
    this.handlers.set(agentName, handler);
  }

  subscribe(taskId: string, listener: (event: TaskEvent) => void) {
    const existing = this.listeners.get(taskId) ?? [];
    existing.push(listener);
    this.listeners.set(taskId, existing);
  }

  async sendMessage(
    agentName: string,
    message: AgentMessage,
    contextId?: string
  ): Promise<Task> {
    const handler = this.handlers.get(agentName);
    if (!handler) {
      const task = this.createTask(contextId);
      task.status = {
        state: "rejected",
        timestamp: Date.now(),
        message: textMessage("agent", `No handler for ${agentName}`),
      };
      return task;
    }

    const task = this.createTask(contextId);
    task.history.push(message);
    task.status = { state: "submitted", timestamp: Date.now() };

    this.processTask(task, handler, message).catch((err) => {
      task.status = {
        state: "failed",
        timestamp: Date.now(),
        message: textMessage("agent", String(err)),
      };
    });
    return task;
  }

  getTask(taskId: string): Task | undefined {
    return this.tasks.get(taskId);
  }

  cancelTask(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || TERMINAL_STATES.includes(task.status.state)) return false;
    task.status = { state: "canceled", timestamp: Date.now() };
    this.emit(taskId, {
      kind: "statusUpdate",
      taskId,
      status: task.status,
    });
    return true;
  }

  private createTask(contextId?: string): Task {
    const task: Task = {
      id: crypto.randomUUID(),
      contextId: contextId ?? crypto.randomUUID(),
      status: { state: "submitted", timestamp: Date.now() },
      artifacts: [],
      history: [],
    };
    this.tasks.set(task.id, task);
    return task;
  }

  private async processTask(
    task: Task,
    handler: TaskHandler,
    message: AgentMessage
  ) {
    task.status = { state: "working", timestamp: Date.now() };
    this.emit(task.id, {
      kind: "statusUpdate",
      taskId: task.id,
      status: task.status,
    });

    try {
      for await (const event of handler(task, message)) {
        if (TERMINAL_STATES.includes(task.status.state)) break;

        if (event.kind === "statusUpdate") {
          task.status = event.status;
        }
        if (event.kind === "artifactUpdate") {
          const existing = task.artifacts.find(
            (a) => a.id === event.artifact.id
          );
          if (existing && event.append) {
            existing.parts.push(...event.artifact.parts);
          } else {
            task.artifacts.push(event.artifact);
          }
        }
        this.emit(task.id, event);
      }
    } catch (err) {
      task.status = {
        state: "failed",
        timestamp: Date.now(),
        message: textMessage("agent", String(err)),
      };
      this.emit(task.id, {
        kind: "statusUpdate",
        taskId: task.id,
        status: task.status,
      });
    }
  }

  private emit(taskId: string, event: TaskEvent) {
    for (const listener of this.listeners.get(taskId) ?? []) {
      listener(event);
    }
  }
}
```

यह वास्तविक A2A कार्य जीवन चक्र को लागू करता हैः प्रस्तुत, काम, इनपुट-आवश्यक, टर्मिनल राज्य। हैंडलर असिनक्रोनस जनरेटर हैं जो SSE स्ट्रीमिंग मॉडल से मेल खाने वाली घटनाओं (स्थिति अपडेट और आर्टिफैक्ट टुकड़े) का उत्पादन करते हैं।

### चरण 4: एसीपी-शैली लेखा परीक्षा पथ

ट्रैकिंग ट्रैकिंग के साथ संचार को शामिल करेंः

```typescript
type AuditEntry = {
  runId: string;
  agentName: string;
  input: AgentMessage[];
  output: AgentMessage[];
  trajectory: TrajectoryEntry[];
  status: "created" | "in-progress" | "completed" | "failed" | "awaiting";
  startedAt: number;
  completedAt?: number;
  sessionId?: string;
};

class AuditableRunner {
  private log: AuditEntry[] = [];
  private handlers: Map<
    string,
    (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  > = new Map();

  registerAgent(
    name: string,
    handler: (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  ) {
    this.handlers.set(name, handler);
  }

  async run(
    agentName: string,
    input: AgentMessage[],
    sessionId?: string
  ): Promise<AuditEntry> {
    const entry: AuditEntry = {
      runId: crypto.randomUUID(),
      agentName,
      input: structuredClone(input),
      output: [],
      trajectory: [],
      status: "created",
      startedAt: Date.now(),
      sessionId,
    };
    this.log.push(entry);

    const handler = this.handlers.get(agentName);
    if (!handler) {
      entry.status = "failed";
      return entry;
    }

    entry.status = "in-progress";
    try {
      const result = await handler(input);
      entry.output = structuredClone(result.output);
      entry.trajectory = structuredClone(result.trajectory);
      entry.status = "completed";
      entry.completedAt = Date.now();
    } catch (err) {
      entry.status = "failed";
      entry.trajectory.push({
        reasoning: `Error: ${String(err)}`,
        timestamp: Date.now(),
      });
      entry.completedAt = Date.now();
    }
    return entry;
  }

  getFullAuditLog(): AuditEntry[] {
    return structuredClone(this.log);
  }

  getAuditLogForAgent(agentName: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.agentName === agentName)
    );
  }

  getAuditLogForSession(sessionId: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.sessionId === sessionId)
    );
  }

  getTrajectoryForRun(runId: string): TrajectoryEntry[] {
    const entry = this.log.find((e) => e.runId === runId);
    return entry ? structuredClone(entry.trajectory) : [];
  }
}
```

प्रत्येक एजेंट निष्पादन एक पूर्ण लेखांकन प्रविष्टि उत्पन्न करता हैः क्या आया, क्या बाहर आया, और उपकरण कॉल और तर्क चरणों के बीच के पूर्ण प्रक्षेपवक्र। आप एजेंट द्वारा, सत्र द्वारा या व्यक्तिगत रन द्वारा पूछ सकते हैं।

### चरण 5: एएनपी-शैली पहचान सत्यापन

डीआईडी आधारित पहचान और सत्यापन का निर्माण करेंः

```typescript
type VerificationMethod = {
  id: string;
  type: string;
  controller: string;
  publicKeyDer: string;
};

type DIDDocument = {
  id: string;
  verificationMethod: VerificationMethod[];
  authentication: string[];
  keyAgreement: string[];
  humanAuthorization: string[];
  service: { id: string; type: string; serviceEndpoint: string }[];
};

type AgentIdentity = {
  did: string;
  document: DIDDocument;
  privateKey: crypto.KeyObject;
  publicKey: crypto.KeyObject;
};

class IdentityRegistry {
  private documents: Map<string, DIDDocument> = new Map();

  publish(doc: DIDDocument) {
    this.documents.set(doc.id, doc);
  }

  resolve(did: string): DIDDocument | undefined {
    return this.documents.get(did);
  }

  verify(did: string, signature: string, payload: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;

    const authKeyIds = doc.authentication;
    const authKeys = doc.verificationMethod.filter((vm) =>
      authKeyIds.includes(vm.id)
    );

    for (const key of authKeys) {
      const publicKey = crypto.createPublicKey({
        key: Buffer.from(key.publicKeyDer, "base64"),
        format: "der",
        type: "spki",
      });
      const isValid = crypto.verify(
        null,
        Buffer.from(payload),
        publicKey,
        Buffer.from(signature, "hex")
      );
      if (isValid) return true;
    }
    return false;
  }

  requiresHumanAuth(did: string, operationKeyId: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;
    return doc.humanAuthorization.includes(operationKeyId);
  }
}

function createIdentity(domain: string, agentName: string): AgentIdentity {
  const did = `did:wba:${domain}:agent:${agentName}`;
  const { publicKey, privateKey } = crypto.generateKeyPairSync("ed25519");

  const publicKeyDer = publicKey
    .export({ format: "der", type: "spki" })
    .toString("base64");

  const keyId = `${did}#key-1`;
  const encKeyId = `${did}#key-x25519-1`;

  const document: DIDDocument = {
    id: did,
    verificationMethod: [
      {
        id: keyId,
        type: "Ed25519VerificationKey2020",
        controller: did,
        publicKeyDer,
      },
      {
        id: encKeyId,
        type: "X25519KeyAgreementKey2019",
        controller: did,
        publicKeyDer,
      },
    ],
    authentication: [keyId],
    keyAgreement: [encKeyId],
    humanAuthorization: [],
    service: [
      {
        id: `${did}#agent-description`,
        type: "AgentDescription",
        serviceEndpoint: `https://${domain}/agents/${agentName}/ad.json`,
      },
    ],
  };

  return { did, document, privateKey, publicKey };
}

function signPayload(identity: AgentIdentity, payload: string): string {
  return crypto
    .sign(null, Buffer.from(payload), identity.privateKey)
    .toString("hex");
}
```

यह वास्तविक एएनपी पहचान मॉडल को दर्शाता हैः एजेंटों के पास अलग-अलग प्रमाणीकरण, कुंजी समझौते और मानव प्राधिकरण कुंजी के साथ डीआईडी दस्तावेज हैं।`IdentityRegistry`डीआईडी रिज़ॉल्यूशन का अनुकरण करता है (उत्पादन में यह एजेंट के डोमेन पर HTTP लाती है) ।

### चरण 6: प्रोटोकॉल गेटवे

सभी चार प्रोटोकॉल को एक एकीकृत प्रणाली में जोड़ेंः

```mermaid
graph LR
    REQ[Incoming Request] --> ANP_V{ANP: Verify DID}
    ANP_V -->|Valid| A2A_D{A2A: Discover Agent}
    ANP_V -->|Invalid| REJECT[Reject]
    A2A_D -->|Found| ACP_A[ACP: Audit Run]
    A2A_D -->|Not Found| REJECT
    ACP_A --> A2A_T[A2A: Create Task]
    A2A_T --> RESULT[Task + Audit Entry]

    style ANP_V fill:#d1fae5,stroke:#059669
    style A2A_D fill:#dbeafe,stroke:#2563eb
    style ACP_A fill:#fef3c7,stroke:#d97706
    style A2A_T fill:#dbeafe,stroke:#2563eb
```

```typescript
class ProtocolGateway {
  private registry: AgentRegistry;
  private taskManager: TaskManager;
  private auditRunner: AuditableRunner;
  private identityRegistry: IdentityRegistry;

  constructor(
    registry: AgentRegistry,
    taskManager: TaskManager,
    auditRunner: AuditableRunner,
    identityRegistry: IdentityRegistry
  ) {
    this.registry = registry;
    this.taskManager = taskManager;
    this.auditRunner = auditRunner;
    this.identityRegistry = identityRegistry;
  }

  async delegateTask(
    fromDid: string,
    signature: string,
    targetAgent: string,
    message: AgentMessage,
    sessionId?: string
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    if (!this.identityRegistry.verify(fromDid, signature, message.id)) {
      return { error: "Identity verification failed" };
    }

    const card = this.registry.resolve(targetAgent);
    if (!card) {
      return { error: `Agent ${targetAgent} not found in registry` };
    }

    const audit = await this.auditRunner.run(
      targetAgent,
      [message],
      sessionId
    );
    const task = await this.taskManager.sendMessage(targetAgent, message);

    return { task, audit };
  }

  discoverAndDelegate(
    fromDid: string,
    signature: string,
    skillTag: string,
    message: AgentMessage
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    const candidates = this.registry.discoverBySkillTag(skillTag);
    if (candidates.length === 0) {
      return Promise.resolve({
        error: `No agents found with skill tag: ${skillTag}`,
      });
    }
    return this.delegateTask(
      fromDid,
      signature,
      candidates[0].name,
      message
    );
  }
}
```

गेटवे एक कॉल में चार चीजें करता हैः
1. **ANP**: DID हस्ताक्षर के माध्यम से कॉल करने वाले की पहचान सत्यापित करता है
2. **A2A**: लक्ष्य एजेंट की खोज करता है और क्षमताओं की जांच करता है
3. **ACP**: ट्रैकटोरियल के साथ एक लेखा परीक्षा पथ में निष्पादन को लपेटता है
4. **A2A**: पूर्ण जीवन चक्र ट्रैकिंग के साथ एक कार्य बनाता है

### चरण 7: इसे एक साथ जोड़ें

```typescript
async function protocolDemo() {
  const registry = new AgentRegistry();
  registry.register({
    name: "researcher",
    description: "Searches and summarizes findings",
    version: "1.0.0",
    url: "https://researcher.local/a2a/v1",
    capabilities: { streaming: true, pushNotifications: false },
    defaultInputModes: ["text/plain"],
    defaultOutputModes: ["text/plain", "application/json"],
    skills: [
      {
        id: "web-research",
        name: "Web Research",
        description: "Searches the web",
        tags: ["research", "search", "summarization"],
        inputModes: ["text/plain"],
        outputModes: ["application/json"],
      },
    ],
  });
  registry.register({
    name: "coder",
    description: "Writes code from specs",
    version: "1.0.0",
    url: "https://coder.local/a2a/v1",
    capabilities: { streaming: false, pushNotifications: false },
    defaultInputModes: ["text/plain", "application/json"],
    defaultOutputModes: ["text/plain"],
    skills: [
      {
        id: "code-gen",
        name: "Code Generation",
        description: "Generates code",
        tags: ["coding", "generation"],
        inputModes: ["text/plain", "application/json"],
        outputModes: ["text/plain"],
      },
    ],
  });

  const taskManager = new TaskManager();
  const auditRunner = new AuditableRunner();

  const researchTrajectory: TrajectoryEntry[] = [];

  taskManager.registerHandler(
    "researcher",
    async function* (task, message) {
      yield {
        kind: "statusUpdate" as const,
        taskId: task.id,
        status: { state: "working" as const, timestamp: Date.now() },
      };

      researchTrajectory.push({
        reasoning: "Searching for React 19 documentation",
        toolName: "web_search",
        toolInput: { query: "React 19 compiler features" },
        toolOutput: {
          results: ["react.dev/blog/react-19", "github.com/react/react"],
        },
        timestamp: Date.now(),
      });

      researchTrajectory.push({
        reasoning: "Extracting key findings from search results",
        toolName: "doc_analysis",
        toolInput: { url: "react.dev/blog/react-19" },
        toolOutput: {
          summary:
            "React 19 compiler auto-memoizes, no manual useMemo needed",
        },
        timestamp: Date.now(),
      });

      yield {
        kind: "artifactUpdate" as const,
        taskId: task.id,
        artifact: {
          id: crypto.randomUUID(),
          name: "research-results",
          parts: [
            {
              kind: "data" as const,
              data: {
                findings: [
                  "React 19 compiler auto-memoizes components",
                  "No more manual useMemo/useCallback needed",
                  "Compiler runs at build time, not runtime",
                ],
                sources: ["react.dev/blog/react-19"],
              },
              mediaType: "application/json",
            },
          ],
        },
        append: false,
        lastChunk: true,
      };

      yield {
        kind: "statusUpdate" as const,
        taskId: task.id,
        status: { state: "completed" as const, timestamp: Date.now() },
      };
    }
  );

  auditRunner.registerAgent("researcher", async () => ({
    output: [
      textMessage("agent", "React 19 compiler auto-memoizes components"),
    ],
    trajectory: researchTrajectory,
  }));

  const identityRegistry = new IdentityRegistry();

  const coderIdentity = createIdentity("coder.local", "coder");
  const researcherIdentity = createIdentity("researcher.local", "researcher");

  identityRegistry.publish(coderIdentity.document);
  identityRegistry.publish(researcherIdentity.document);

  const gateway = new ProtocolGateway(
    registry,
    taskManager,
    auditRunner,
    identityRegistry
  );

  console.log("=== Protocol Demo ===\n");

  console.log("1. Agent Discovery (A2A)");
  const researchAgents = registry.discoverBySkillTag("research");
  console.log(
    `   Found ${researchAgents.length} agent(s):`,
    researchAgents.map((a) => a.name)
  );

  console.log("\n2. Identity Verification (ANP)");
  const message = textMessage("user", "Research React 19 compiler features");
  const signature = signPayload(coderIdentity, message.id);
  const verified = identityRegistry.verify(
    coderIdentity.did,
    signature,
    message.id
  );
  console.log(`   Coder DID: ${coderIdentity.did}`);
  console.log(`   Signature verified: ${verified}`);

  console.log("\n3. Task Delegation (A2A + ACP + ANP)");
  const result = await gateway.delegateTask(
    coderIdentity.did,
    signature,
    "researcher",
    message,
    "session-001"
  );

  if ("error" in result) {
    console.log(`   Error: ${result.error}`);
    return;
  }

  console.log(`   Task ID: ${result.task.id}`);
  console.log(`   Task state: ${result.task.status.state}`);
  console.log(`   Artifacts: ${result.task.artifacts.length}`);

  console.log("\n4. Audit Trail (ACP)");
  console.log(`   Run ID: ${result.audit.runId}`);
  console.log(`   Status: ${result.audit.status}`);
  console.log(`   Trajectory steps: ${result.audit.trajectory.length}`);
  for (const step of result.audit.trajectory) {
    console.log(`     - ${step.reasoning}`);
    if (step.toolName) {
      console.log(`       Tool: ${step.toolName}`);
    }
  }

  console.log("\n5. Full Audit Log");
  const fullLog = auditRunner.getFullAuditLog();
  console.log(`   Total runs: ${fullLog.length}`);
  for (const entry of fullLog) {
    const duration = entry.completedAt
      ? `${entry.completedAt - entry.startedAt}ms`
      : "in-progress";
    console.log(`   ${entry.agentName}: ${entry.status} (${duration})`);
  }
}

protocolDemo().catch((err) => {
  console.error("Protocol demo failed:", err);
  process.exitCode = 1;
});
```

## क्या गलत हो रहा है

प्रोटोकॉल खुश पथ को हल करते हैं।

**Schema drift.**एजेंट ए एक एजेंट कार्ड विज्ञापन प्रकाशित करता है `application/json`आउटपुट. लेकिन JSON योजना संस्करणों के बीच बदलता है. एजेंट बी पुराने प्रारूप को विश्लेषण करता है और कचरा प्राप्त करता है. फिक्सः संस्करण अपने कौशल और आउटपुट योजनाओं. ए 2 ए विनिर्देश समर्थन करता है `version`एजेंट कार्ड्स पर इस कारण से.

**State machine violations.**एक एजेंट हैंडल एक `completed`घटना, फिर अधिक कलाकृतियों को उत्पन्न करने की कोशिश करता है। कार्य अपरिवर्तनीय है। आपका कोड चुपचाप अद्यतन छोड़ देता है या फेंक देता है। ठीकः उत्पन्न करने से पहले टर्मिनल स्थिति की जांच करें। `TaskManager`उपरोक्त के साथ यह लागू करता है `break`टर्मिनल राज्यों के बाद।

**Trust resolution failures.**एजेंट ए एजेंट बी के डीआईडी की पुष्टि करने की कोशिश करता है, लेकिन एजेंट बी का डोमेन डाउन है। डीआईडी दस्तावेज़ नहीं लाया जा सकता है। क्या आप खोलने में विफल रहते हैं (अनवीरीफाई किए गए एजेंटों को स्वीकार करते हैं) या बंद करने में विफल रहते हैं (सब कुछ अस्वीकार करते हैं)? एएनपी कम से कम विश्वास के सिद्धांत के साथ बंद करने में विफल रहने की सिफारिश करता है।

**Trajectory bloat.**एसीपी ट्रैक्टरी लॉगिंग शक्तिशाली है लेकिन महंगा है। एक जटिल एजेंट जो प्रति रन 200 टूल कॉल करता है, विशाल ऑडिट प्रविष्टियां उत्पन्न करता है। फिक्सः कॉन्फ़िगर करने योग्य वर्बोसिटी स्तरों पर लॉग ट्रैक्टरी। अनुपालन के लिए टूल नाम और आईओ रिकॉर्ड करें, गैर-नियामक कार्यभार के लिए तर्क चरणों को छोड़ दें।

**Discovery thundering herd.**50 एजेंट सभी पूछताछ `GET /agents`स्टार्टअप पर एक साथ. ठीकः TTL के साथ कैश एजेंट कार्ड, स्टेजिंग डिस्कवरी अंतराल, या मतदान के बजाय पुश आधारित पंजीकरण का उपयोग करें।

## इसका प्रयोग करें

### वास्तविक कार्यान्वयन

**A2A**सबसे परिपक्व है. गूगल की [official spec](https://github.com/google/A2A)यदि आपके एजेंटों को गतिशील खोज और सहयोग की आवश्यकता है, तो यहां से शुरू करें।

**ACP**आईबीएम के [BeeAI project](https://github.com/i-am-bee/acp)एसीपी (trajectory logging, run life cycle) का उपयोग करें, भले ही आप ए 2 ए का उपयोग परिवहन के रूप में करें।

**ANP**यह सबसे प्रयोगात्मक है।[community repo](https://github.com/agent-network-protocol/AgentNetworkProtocol)एक पायथन एसडीके (एजेंट कनेक्ट) है। मेटा-प्रोटोकॉल बातचीत अवधारणा वास्तव में नया है। क्रॉस-संगठन एजेंट तैनाती के लिए देखने लायक है।

**MCP**यदि आप एजेंटों को उपकरण का उपयोग करना चाहते हैं, तो एमसीपी मानक है।

### सही प्रोटोकॉल चुनना

```mermaid
graph TD
    START{Do agents need<br/>to use tools?}
    START -->|Yes| MCP_R[Use MCP]
    START -->|No| TALK{Do agents need to<br/>talk to each other?}
    TALK -->|No| NONE[You don't need<br/>a protocol]
    TALK -->|Yes| AUDIT{Need audit trails<br/>for compliance?}
    AUDIT -->|Yes| ACP_R[A2A + ACP<br/>trajectory patterns]
    AUDIT -->|No| ORG{All agents<br/>within your org?}
    ORG -->|Yes| A2A_R[A2A<br/>Agent Cards + Tasks]
    ORG -->|No| INFRA{Shared<br/>infrastructure?}
    INFRA -->|Yes| BROKER[A2A + message broker]
    INFRA -->|No| ANP_R[ANP + A2A<br/>DID verification]

    style MCP_R fill:#d1fae5,stroke:#059669
    style A2A_R fill:#dbeafe,stroke:#2563eb
    style ACP_R fill:#fef3c7,stroke:#d97706
    style ANP_R fill:#f3e8ff,stroke:#7c3aed
    style BROKER fill:#e0e7ff,stroke:#4338ca
```

## इसे भेजें

इस पाठ से उत्पन्न होता हैः
- `code/main.ts`-- चार प्रोटोकॉल पैटर्नों का पूर्ण कार्यान्वयन
- `outputs/prompt-protocol-selector.md`-- एक संकेत जो आपको अपने सिस्टम के लिए प्रोटोकॉल चुनने में मदद करता है

## व्यायाम

1. **Multi-hop task delegation.**`TaskManager`एक एजेंट हैंडलर अन्य एजेंटों को सबटास्क सौंप सकता है। शोधकर्ता को एक कार्य प्राप्त होता है, दो विशेषज्ञ एजेंटों को सबटास्क "खोज" और "संक्षेप" देता है, दोनों को पूरा होने की प्रतीक्षा करता है, फिर परिणामों को अपने स्वयं के कलाकृतियों में मिलाता है।

2. **Streaming audit trail.** को संशोधित करें`AuditableRunner`पूर्ण परिणाम की प्रतीक्षा करने के बजाय, yield `AuditEntry`वास्तविक समय में अद्यतन के रूप में प्रक्षेपवक्र प्रविष्टियों को जोड़ा जाता है. एक async जनरेटर का उपयोग जो ऑडिट स्नैपशॉट उत्पन्न करता है।

3. **DID rotation.** में कुंजी घूर्णन जोड़ें`IdentityRegistry`एक एजेंट को एक अद्यतन कुंजी के साथ एक नया डीआईडी दस्तावेज़ प्रकाशित करने में सक्षम होना चाहिए जबकि एक `previousDid`प्रमाणिकरणकर्ता को एक समय अवधि के दौरान वर्तमान और पूर्व कुंजी दोनों से हस्ताक्षर स्वीकार करना चाहिए।

4. **Protocol negotiation.**एएनपी के मेटा प्रोटोकॉल अवधारणा को लागू करें। दो एजेंटों का आदान-प्रदान करें।`protocolNegotiation`उम्मीदवार प्रारूपों के साथ संदेश (जैसे, "मैं JSON-RPC बोल सकता हूं" बनाम "मैं REST पसंद करता हूं") अधिकतम 3 राउंड के बाद, वे एक प्रारूप या टाइमआउट पर सहमत होते हैं। सहमत प्रारूप निर्धारित करता है कि कौन सा `TaskManager`या `AuditableRunner`वे उपयोग करते हैं.

5. **Rate-limited discovery.**एक जोड़ें `RateLimitedRegistry`एक wrapper जो एक कॉन्फ़िगर करने योग्य TTL के साथ एजेंट कार्ड खोजों को कैश करता है और प्रति एजेंट प्रति सेकंड खोज प्रश्नों को सीमित करता है। स्टार्टअप पर एक दूसरे को खोजने वाले 100 एजेंटों के एक थंडर झुंड का अनुकरण करें और अंतर को मापें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| MCP | "The protocol for AI tools" | A client-server protocol for agents to discover and use tools. Agent-to-tool, not agent-to-agent. |
| A2A | "Google's agent protocol" | A peer-to-peer protocol for agent collaboration under the Linux Foundation. Discovery via Agent Cards, 9-state task lifecycle, streaming via SSE. Supports JSON-RPC, REST, and gRPC bindings. |
| ACP | "Enterprise agent messaging" | IBM/BeeAI's REST API for agent runs with TrajectoryMetadata: every response carries the full chain of reasoning and tool calls. Merging into A2A. |
| ANP | "Decentralized agent identity" | A community protocol using `did:wba` (DID) for cryptographic identity, HPKE for E2EE, and AI-powered meta-protocol negotiation for agents that have never seen each other. |
| Agent Card | "An agent's business card" | A JSON document at `/.well-known/agent-card.json` describing skills, supported MIME types, security schemes, and protocol bindings. |
| DID | "Decentralized ID" | W3C standard for cryptographically verifiable identities hosted on the agent's own domain. ANP uses `did:wba` method. |
| TrajectoryMetadata | "The audit receipt" | ACP's mechanism for attaching reasoning steps, tool calls, and their inputs/outputs to every agent response. |
| Meta-protocol | "Agents negotiating how to talk" | ANP's approach where agents use natural language to dynamically agree on data formats, then generate code to handle them. |
| Task | "A unit of work" | A2A's stateful object tracking work from submission through completion. Immutable once terminal. |

## आगे पढ़ना

- [Google A2A specification](https://github.com/google/A2A)-- आधिकारिक विनिर्देश और SDKs (v1.0.0, लिनक्स फाउंडेशन)
- [IBM/BeeAI ACP specification](https://github.com/i-am-bee/acp)-- एजेंट रन और ट्रैकटोरिया मेटाडेटा के लिए ओपनएपीआई 3.1 विनिर्देश
- [Agent Network Protocol](https://github.com/agent-network-protocol/AgentNetworkProtocol)-- डीआईडी आधारित पहचान, ई2ईई, मेटा प्रोटोकॉल बातचीत
- [Model Context Protocol docs](https://modelcontextprotocol.io/)-- एंथ्रोपिक के एमसीपी विनिर्देश (चरण 13 में शामिल)
- [W3C Decentralized Identifiers](https://www.w3.org/TR/did-core/)-- एएनपी के आधार पर पहचान मानक
- [RFC 9180 (HPKE)](https://www.rfc-editor.org/rfc/rfc9180)-- एनएपी E2EE के लिए उपयोग की जाने वाली एन्क्रिप्शन योजना
- [FIPA Agent Communication Language](http://www.fipa.org/specs/fipa00061/SC00061G.html)-- आधुनिक एजेंट प्रोटोकॉल के लिए अकादमिक अग्रदूत
