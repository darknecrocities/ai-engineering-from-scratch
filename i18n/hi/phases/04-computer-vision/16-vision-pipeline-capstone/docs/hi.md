# एक पूर्ण दृष्टि पाइपलाइन का निर्माण करें  कैपस्टोन

> एक उत्पादन दृष्टि प्रणाली मॉडल और नियमों की एक श्रृंखला है जो डेटा अनुबंधों से सिलाई जाती है। टुकड़े पहले से ही इस चरण में हैं; कैपस्टोन उन्हें अंत से अंत तक एक साथ जोड़ता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lessons 01-15
**Time:** ~120 minutes

## सीखने के लक्ष्य

- एक उत्पादन दृष्टि पाइपलाइन डिजाइन करें जो वस्तुओं का पता लगाता है, उन्हें वर्गीकृत करता है, और प्रत्येक विफलता पथ के साथ संरचित JSON  उत्सर्जित करता है
- एक डिटेक्टर (मास्क आर-सीएनएन या यॉलॉ), एक वर्गीकरणकर्ता (ConvNeXt-Tiny) और एक डेटा अनुबंध (पायदान्टिक) को एक सेवा में प्लग करें
- अंत-से-अंत पाइपलाइन का बेंचमार्क करें और पहला बोतल गला (आमतौर पर पूर्व प्रसंस्करण, फिर डिटेक्टर) की पहचान करें
- एक न्यूनतम FastAPI सेवा भेजें जो छवि अपलोड को स्वीकार करता है, पाइपलाइन चलाता है, और वर्गीकरण के साथ पता लगाने को लौटाता है

## समस्या

व्यक्तिगत दृष्टि मॉडल उपयोगी हैं; दृष्टि उत्पाद उनकी श्रृंखलाएं हैं। एक खुदरा शेल्फ ऑडिट एक डिटेक्टर प्लस एक उत्पाद वर्गीकरण प्लस एक मूल्य-ओसीआर पाइपलाइन है। स्वायत्त ड्राइविंग एक 2 डी डिटेक्टर प्लस एक 3 डी डिटेक्टर प्लस एक सेगमेंटर प्लस एक ट्रैकर प्लस एक प्लानर है। एक चिकित्सा प्री-स्क्रीन एक सेगमेंटर प्लस एक क्षेत्र वर्गीकरण प्लस एक क्लिनिक UI है।

इन श्रृंखलाओं को तार करना वह हिस्सा है जो एक एमएल प्रोटोटाइप को उत्पाद से अलग करता है। मॉडल के बीच प्रत्येक इंटरफ़ेस बग के लिए एक नया स्थान है। प्रत्येक निर्देशांक परिवर्तन, प्रत्येक सामान्यीकरण, प्रत्येक मास्क आकार परिवर्तन एक चुप्पी विफलता उम्मीदवार है। एक पाइपलाइन अपने सबसे कमजोर इंटरफ़ेस के रूप में मजबूत है।

इस कैपस्टोन में न्यूनतम व्यवहार्य पाइपलाइन सेट की जाती हैः पता लगाने + वर्गीकरण + संरचित आउटपुट + एक सेवा परत। इस कंकाल में चरण 4 स्लॉट में बाकी सब कुछः YOLOv8 के लिए मास्क आर-सीएनएन को स्वैप करें, एक ओसीआर हेड जोड़ें, एक खंड शाखा जोड़ें, एक ट्रैकर जोड़ें। वास्तुकला स्थिर है; टुकड़े प्लग करने योग्य हैं।

## अवधारणा

### पाइपलाइन

```mermaid
flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detections + classes"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a
```

दो मॉडल चरण महंगे हैं, जबकि पांच अन्य चरणों में कीड़े रहते हैं।

### Pydantic के साथ डेटा अनुबंध

प्रत्येक मॉडल सीमा एक टाइप वस्तु बन जाती है, जिससे चुपचाप विफलताएं जोर से होती हैं।

```
Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)
```

जब एक डिटेक्टर बॉक्स वापस `(cx, cy, w, h)`इसके बजाय `(x1, y1, x2, y2)`, Pydantic की सत्यापन सीमा पर विफल रहता है और आप तुरंत पता लगाने के बजाय एक डाउनस्ट्रीम फसल डिबग जो चुपचाप रिक्त क्षेत्रों को वापस देता है।

### जहां विलंबता जाती है

लगभग हर दृष्टि पाइपलाइन में तीन सत्य हैं:

1. **Preprocessing is often the biggest single block.**JPEG को डिकोड करना, रंग स्थानों को परिवर्तित करना, आकार बदलना  ये CPU-bound हैं और भूलना आसान है।
2. **The detector dominates GPU time.**70-90% GPU समय पता लगाने आगे पास में है।
3. **Postprocessing (NMS, RLE encode/decode) is cheap on GPU, expensive on CPU.**हमेशा वास्तविक लक्ष्य के साथ प्रोफ़ाइल.

वितरण को जानना ही अनुकूलन को प्राथमिकता सूची में बदल देता है।

### विफलता मोड

- **Empty detections** रिक्त सूची लौटाएं, दुर्घटनाग्रस्त न हों। लॉग.
- **Out-of-bounds boxes** कटौती से पहले छवि आकार को क्लैंप करें।
- **Tiny crops** वर्गीकरण के न्यूनतम इनपुट से छोटे बॉक्स के लिए वर्गीकरण छोड़ दें।
- **Corrupt upload** 400 प्रतिक्रिया के साथ एक विशिष्ट त्रुटि कोड, 500 नहीं।
- **Model load failure** सेवा शुरू करने पर विफलता, पहले अनुरोध पर नहीं।

एक उत्पादन पाइपलाइन इन सभी को बिना सामान्य लेखन के संभालती है `try/except`हर असफलता को एक नामित कोड और एक प्रतिक्रिया मिलती है।

### बैचिंग

एक उत्पादन सेवा कई ग्राहकों की सेवा करती है। अनुरोधों के बीच बैचिंग डिटेक्शन और वर्गीकरण पारगमन को गुणा करता है। व्यापारः एक बैच भरने की प्रतीक्षा करने से अतिरिक्त विलंबता। विशिष्ट सेटअपः 20ms तक के अनुरोध एकत्र करें, बैच एक साथ करें, प्रक्रिया करें, प्रतिक्रियाएं वितरित करें। `torchserve`और `triton`यह मूल रूप से करें; अनुमानित लोड के साथ छोटी सेवाएं अपने स्वयं के माइक्रो-बैचर को रोल करें।

```figure
v4-vision-pipeline
```

## इसे बनाओ

### चरण 1: डेटा अनुबंध

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

कोड के पांच सेकंड किसी भी गंभीर पाइपलाइन पर डिबगिंग के एक घंटे बचाता है।

### चरण 2: न्यूनतम पाइपलाइन वर्ग

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

प्रत्येक इंटरफ़ेस टाइप किया जाता है. प्रत्येक विफलता पथ एक विशिष्ट हैंडलिंग निर्णय है.

### चरण 3: एक डिटेक्टर और एक वर्गीकरण तार

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# Use ImageNet-pretrained weights for a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### चरण 4: फास्टएपीआई सेवा

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialised on startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

दौड़ो`uvicorn main:app --host 0.0.0.0 --port 8000`. परीक्षण के साथ `curl -F 'file=@dog.jpg' http://localhost:8000/detect`. .

### चरण 5: पाइपलाइन को बेंचमार्क करें

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warm up

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

CPU पर विशिष्ट आउटपुटः प्रीप्रोसेस ~3 ms, डिटेक्ट 300-500 ms, वर्गीकृत 20-40 ms, कुल 350-550 ms। GPU पर, डिटेक्ट 20-40 ms है और प्रीप्रोसेस + वर्गीकृत सापेक्ष रूप से अधिक मायने रखता है।

## इसका प्रयोग करें

उत्पादन टेम्पलेट एक ही संरचना में अभिसरण करते हैं, और इसके अतिरिक्तः

- **Model versioning** हमेशा उत्तर में मॉडल नाम और वजन हैश को लॉग करें।
- **Per-request trace IDs** प्रत्येक अनुरोध के लिए प्रत्येक चरण का समय रिकॉर्ड करें ताकि आप धीमी प्रतिक्रियाओं को चरणों के साथ जोड़ सकें।
- **Fallback path** यदि वर्गीकरणकर्ता समय समाप्त हो जाता है, तो पूरी मांग को विफल करने के बजाय बिना वर्गीकरण के पता लगाने को लौटाएं।
- **Safety filters** NSFW/PII फ़िल्टर वर्गीकरण के बाद, प्रतिक्रिया सेवा से बाहर जाने से पहले चलाया जाता है।
- **Batch endpoint** एक `/detect_batch`बड़े पैमाने पर प्रसंस्करण के लिए छवि URL की सूची स्वीकार करना।

उत्पादन सेवा के लिए, `torchserve`,`Triton Inference Server`और `BentoML`बैचिंग, संस्करण, मीट्रिक, और स्वास्थ्य जांच को संभालते हैं।`FastAPI`सीधे प्रोटोटाइप और छोटे पैमाने पर उत्पादों के लिए ठीक है।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः

- `outputs/prompt-vision-service-shape-reviewer.md` एक संकेत जो अनुबंध/उत्तर आकार उल्लंघन के लिए विजन सेवा के कोड की समीक्षा करता है और पहले टूटने वाले बग का नाम देता है।
- `outputs/skill-pipeline-budget-planner.md` एक कौशल जो लक्ष्य विलंबता और पारगम्यता को देखते हुए, पाइपलाइन के प्रत्येक चरण के लिए एक समय बजट आवंटित करता है और यह दर्शाता है कि किस चरण को अपना बजट सबसे पहले याद आएगा।

## व्यायाम

1. **(Easy)**किसी भी खुले डेटासेट से 10 छवियों पर पाइपलाइन चलाएं। प्रति चरण औसत समय और प्रति छवि पता लगाने की गणना का वितरण रिपोर्ट करें।
2. **(Medium)** में मुखौटा आउटपुट फ़ील्ड जोड़ें`Detection`JSON 10 वस्तुओं के लिए भी 1MB से कम रहता है सत्यापित करें।
3. **(Hard)**वर्गीकरणकर्ता के सामने एक माइक्रो-बैचर जोड़ेंः 10 एमएस तक फसलें एकत्र करें, उन्हें एक GPU कॉल में सभी वर्गीकृत करें, प्रति अनुरोध परिणाम वापस करें। प्रति सेकंड 5 समवर्ती अनुरोधों पर संचलन वृद्धि और अतिरिक्त विलंबता मापें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pipeline | "The system" | An ordered chain of preprocessing, inference, and postprocessing steps with a typed interface between each pair |
| Data contract | "The schema" | Pydantic / dataclass definitions that every stage input and output conforms to; catches integration bugs at the boundary |
| Preprocessing | "Before the model" | Decoding, colour conversion, resizing, normalising; usually the biggest CPU time sink |
| Postprocessing | "After the model" | NMS, mask resize, threshold, RLE encode; cheap on GPU, expensive on CPU |
| Microbatcher | "Collect then forward" | Aggregator that waits a fixed window for multiple requests, runs a single batched forward pass |
| Trace ID | "Request id" | Per-request identifier logged at every stage so slow requests can be traced end-to-end |
| Failure code | "Named error" | Specific error code per failure class instead of generic 500; enables client retry logic |
| Health check | "Readiness probe" | Cheap endpoint that reports whether the service can answer; loadbalancers rely on this |

## आगे पढ़ना

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) उत्पादन एमएल तैनाती का कैनोनिक अवलोकन
- [BentoML docs](https://docs.bentoml.com) बैचिंग, वर्शनिंग और मेट्रिक्स के साथ सेवा ढांचे
- [torchserve docs](https://pytorch.org/serve/) पायटॉर्च की आधिकारिक सेवा पुस्तकालय
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) बैचिंग और मल्टी-मॉडल समर्थन के साथ उच्च आउटपुट सेवा
