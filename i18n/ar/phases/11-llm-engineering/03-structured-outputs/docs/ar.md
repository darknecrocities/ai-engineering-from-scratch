# الخروج المهيكل: JSON، تصحيح الخطة، تشريح القيود المحدود

> يعيد ماجستيرك في التدريس سلسلة. تطبيقك يحتاج إلى JSON. هذا الفجوة قد حطمت أكثر من أنظمة الإنتاج من أي الهلوسة النموذجية. الخروج المهيكلي هو الجسر بين اللغة الطبيعية والبيانات المطبوعة. الحصول على الحق و ماجستيرك في التدريس يصبح API موثوقة. الحصول على خطأ و كنت تحليل النص الحر مع regex في 3 صباحا.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 minutes
**Related:**المرحلة 5 · 20 (المخرجات المهيكلة والتشفير المحدود) تغطي نظرية مستوى المفكّر (معالجات FSM / CFG logit ، الخطوط المخططية ، XGrammar). يركز هذا الدروس على سطح SDK الإنتاج (OpenAI `response_format`(استخدام أدوات الأنثروبية، مدرب)  اقرأ المرحلة 5 · 20 أولاً إذا كنت تريد فهم ما يحدث تحت API.

## أهداف التعلم

- تنفيذ خروجيات في وضع JSON ومقيدة من مخطط باستخدام معايير OpenAI و API Anthropic
- بناء طبقة التحقق من Pydantic التي ترفض نتائج LLM الملفوفة وتجربات إعادة مع رد فعل الخطأ
- شرح كيفية قيود فك التشفير القوى JSON صالحة على مستوى الرمز دون بعد المعالجة
- تصميم أجهزة استخراج قوية تحويل النص غير المهيكلي بشكل موثوق به إلى هيكلات بيانات مدرجة

## المشكلة

تسأل ماجستير في العلوم: "استخرج اسم المنتج، السعر، والتوفر من هذا النص".

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

هذا إجابة صحيحة تماماً. إنه لا فائدة له أيضاً لتطبيقك. نظامك للمخزون يحتاج`{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`تحتاج إلى جسم JSON مع مفاتيح محددة وأنواع محددة، وقيود قيمة محددة. لا تحتاج إلى جملة.

الحل البديل: إضافة "رد في JSON" إلى طلبك. هذا يعمل 90٪ من الوقت. 10% الآخر من النموذج يلف JSON في حواجز رمز التسجيل، أو يضيف مقدمة مثل "هنا JSON:"، أو ينتج JSON غير صالحة عن طريق النصية لأنه أغلق شريط مبكرا. محلل JSON الخاص بك ينهار. أنابيبك تتعطل تضيف محاولة/إستثناء و حلقة محاولة ثانية في بعض الأحيان، الإعادة تُنتج بيانات مختلفة. الآن لديك مشكلة التماسك فوق مشكلة التحليل

هذه ليست مشكلة هندسية سريعة. إنها مشكلة فك تشفير. النموذج يولد رموز من اليسار إلى اليمين. في كل موقف، فإنه يختار رمزاً آخر من المفردات من خيارات 100K +. معظم هذه الخيارات ستنتج JSON غير صالحة في أي موقف معين. إذا كان النموذج قد أصدرت للتو `{"price":`, يجب أن تكون الرمز التالي رقم ، اقتباس (لسلسلة) ،`null`،`true`،`false`أي شيء آخر ينتج JSON غير صالحة. بدون قيود، قد يختار النموذج كلمة إنجليزية معقولة تماماً

## المفهوم

### الطيف المُنشئ للإنتاج

هناك أربع مستويات من التحكم في الخروج المهيكلة، كل منها أكثر موثوقية من الأخيرة.

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**Prompt-based**("رد في JSON صالحة"): لا تطبيق. النموذج يلتزم عادة ولكن في بعض الأحيان لا. موثوقية: ~ 90٪. وضع الفشل: حواجز التسجيل، نص المبادئ، الناتج المختصر، بنية خاطئة.

**JSON mode**: يضمن API أن المخرج هو JSON صالح.`response_format: { type: "json_object" }`المخرج سوف تحلل دون أخطاء. ولكن قد لا تتطابق النظام المتوقع -- المفاتيح الإضافية، النوع الخطأ، الحقول المفقودة.

**Schema mode**: يأخذ API مخطط JSON ويضمن أن الخروج يطابقها. في عام 2026 كل مزود رئيسي يدعم هذا بشكل أصلي: OpenAI `response_format: { type: "json_schema", json_schema: {...} }`(و أيضاً)`tool_choice="required"`), استخدام أدوات الأنثروبيك مع `input_schema`، و التوأم`response_schema`+ `response_mime_type: "application/json"`.المخرج لديه المفاتيح والأنواع والقيود التي حددتها بالضبط.

**Constrained decoding**في كل وضع رمز أثناء توليد، يقوم المفكّر بتخفيض جميع الرموز التي ستنتج خروجاً غير صالحة. إذا كانت النموذج تتطلب رقمًا والنموذج على وشك إصدار حرفًا، يتم تعيين هذه الرموز إلى احتمال صفر. يمكن للنموذج إنتاج الرموز التي تؤدي فقط إلى خروجاً صالحًا. هذا ما ينفذه وضع الخروج المهيكلي OpenAI والمكتبات مثل الخطوط المخطوطة والإرشادات تحت الغطاء.

### مخطط JSON: لغة العقد

مخطط JSON هو كيفية إخبار النموذج (أو طبقة التحقق) ما الشكل الذي يجب أن يكون له الخروج. كل نظام خروج مهيكلي كبير يستخدمها.

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

هذا النظام يقول: إنتاج يجب أن يكون كائن مع سلسلة `product`، رقم غير سلبي `price`، و بوليان `in_stock`، وسلسلة اختيارية من السلاسل `categories`أي إصدار لا يطابق يتم رفضه

النماذج تتعامل مع الحالات الصعبة: الأشياء المتعقدة، المجموعات التي تحتوي على عناصر منخفضة، والإينومات (تقييد سلسلة إلى قيم محددة) ، ومطابقة الأنماط (الرد على السلاسل) ، والمزيج (oneOf، anyOf، allOf للمخرجات متعددة الأشكال).

### النمط البيدانتي

في بايثون، لا تكتب مخطط JSON يدوياً. أنت تعريف نموذج Pydantic و يخلق مخطط لك.

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

هذا ينتج نفس مخطط JSON كما هو أعلاه. تقوم مكتبة المعلم (و SDK OpenAI) بقبول نماذج Pydantic مباشرة: اجتياز فئة النموذج، والعودة إلى مثال معتمد. إذا لم تتطابق نتائج LLM، يقوم المعلم بمحاولة إعادة تلقائيًا.

### مكالمة الوظيفة / استخدام الأدوات

واجهة بديلة لنفس المشكلة. بدلاً من طلب النموذج من إنتاج JSON مباشرة، تعريف "الأدوات" (الظروف) مع المعلمات المكتوبة. النموذج يخرج دعوة وظيفة مع حجج مهيكلة. OpenAI يطلق على هذا "دعوة الوظيفة". Anthropic يطلق عليه "استخدام الأدوات". النتيجة هي نفسها: بيانات مهيكلة.

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

يفضل استخدام الأداة عندما يحتاج النموذج إلى اختيار الوظيفة التي سيتم استدعائها ، وليس فقط ملء المعايير. إذا كان لديك 10 مخططات استخراج مختلفة ويجب على النموذج اختيار المناسب على أساس المدخلات ، فإن استخدام الأداة يعطيك كل من اختيار مخطط والمخرج المهيكلي.

### أساليب الفشل الشائعة

حتى مع تنفيذ النظام، الخروج المهيكلة يمكن أن تفشل بطرق خفيفة.

**Hallucinated values**: الخروج يطابق النظام ولكن يحتوي على بيانات مخترعة. النموذج ينتج `{"price": 299.99}`عندما يقول النص 348 دولار. التحقق من النظام لا يمكن أن تلتقط هذا -- النوع صحيح، القيمة خاطئة.

**Enum confusion**: تحدّد حقلًا إلى `["in_stock", "out_of_stock", "preorder"]`. النموذج الخروج`"available"`-- صحيحة من الناحية النطاقية، ولكن ليس في مجموعة المسموح بها. التشخيص المقيد جيد يمنع هذا. نهج مبني على اللحظة لا يفعل ذلك.

**Nested object depth**: النماذج المرتبطة بعمق (4+ مستويات) تنتج أخطاء أكثر. كل مستوى من مستويات التجميد هو مكان آخر حيث يمكن أن يفقد النموذج البنية.

**Array length**: قد ينتج النموذج الكثير جداً أو عدد قليل جداً من المواد في صف.`minItems`و`maxItems`لكن ليس كل مقدمي الخدمات يطبقونها على مستوى فك الكود.

**Optional field omission**: النموذج يفتقر إلى الحقول التي هي اختياريًا تقنيًا ولكن مهمة من الناحية الدلالية لحالة الاستخدام الخاصة بك. حددها حسب المتطلبات في النموذج حتى لو كانت البيانات مفقودة أحيانًا - اجبر النموذج على إنتاج`null`صراحة

```figure
mx-schema-funnel
```

## بناءها

### الخطوة 1: مؤكدة مخطط JSON

قم ببناء مؤكدة من الصفر التي تحقق ما إذا كان كائن Python يطابق مخطط JSON. هذا ما يعمل على جانب الخروج للتحقق من الامتثال.

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### الخطوة الثانية: نموذج في النمط البيانتيكي إلى الخطة

قم ببناء محول من الفئة إلى الخطة الحد الأدنى. حدد فئة Python وولد مخطط JSON تلقائيا.

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### الخطوة الثالثة: تصفية الوسائل المعيارية المحدودة

محاكاة تشفير القيود المحدود. مع إعطاء سلسلة JSON جزئية ونظام رسمية، تحديد أي فئات رمزية صالحة في الموقف الحالي.

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### الخطوة الرابعة: خط أنابيب الاستخراج

قم بتجميع كل شيء في خط أنابيب استخراج: حدد مخططًا، وقلل من LLM ينتج نتائجًا مهيكلةً، وصدّق الناتج، واعتدّل التجربات المُجدّدة.

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### الخطوة 5: قم بتشغيل خط الأنابيب الكامل

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## استخدمها

### المخرجات المهيكلة OpenAI

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

يستخدم وضع الخروج المهيكلي من OpenAI تشفير مقيد داخليا. كل رمز يولد من النموذج يضمن إنتاج خروج يطابق مخطط Pydantic. لا حاجة إلى إعادة التجربة. لا حاجة إلى التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق.

### استخدام الأدوات الإنسانية

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

يصل Anthropic إلى إنتاج مهيكلي من خلال استخدام الأدوات. ينشر النموذج نداءً أدواتًا مع حجج مهيكلية تتطابق مع input_schema. نفس النتيجة ، سطح API مختلف.

### مكتبة المعلمين

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

يقوم المعلم بتصوير أي عميل LLM ويزيد من التجربات التلقائية مع التحقق. إذا فشلت المحاولة الأولى في التحقق، فإنه يرسل الأخطاء إلى النموذج كسياق ويطلب منه إصلاح الخروج. يعمل هذا مع أي مزود، وليس فقط OpenAI.

## أرسله

هذا الدرس يُنتج`outputs/prompt-structured-extractor.md`-- نموذج استقالة قابل للاستعمال الذي يستخرج البيانات المهيكلة من أي نص مع تعريف مخطط. إطعام مخطط JSON والنص غير المهيكلة، ويعود JSON الموثق.

كما أنها تنتج`outputs/skill-structured-outputs.md`-- إطار قرار لانتخاب استراتيجية الخروج المنظمة المناسبة بناء على مزودك، متطلبات موثوقية، وتعقيد مخطط.

## التمارين

1. تمديد مؤكدة النظام لدعم `oneOf`(يجب أن تتطابق البيانات بالضبط مع واحدة من العديد من الخرائط) هذا يتعامل مع المخرجات متعددة الأشكال - على سبيل المثال، حقل يمكن أن يكون إما`Product`أو`Service`كائن ذو أشكال مختلفة

2. قم ببناء أداة "مختلفة الخطة" التي تقارن مخططين وتحدد التغييرات المكسورة (إزالة الحقول المطلوبة، وتغيير الأنواع) مقابل التغييرات غير المكسورة (إضافة الحقول الاختيارية، القيود المسترخة). هذا أمر ضروري لتطبيق مخططات الاستخراج الخاصة بك في الإنتاج.

3. قم بتنفيذ محاكاة تشخيص القيود المحدودة الأكثر واقعية. مع إعطاء مخطط JSON ومجموعة من 100 رمز (حروف وأرقام ونقاط، كلمات رئيسية) ، قم بمشي من خلال إنتاج خطوة بخطوة، وتخفي رموز غير صالحة في كل موقع. قياس مدى نسبة المئة من المفردات صالحة في كل خطوة.

4. قم ببناء مجموعة تقييم الاستخراج. قم بإنشاء 50 وصف للمنتج مع نتائج JSON المسموحة يدوياً. قم بتشغيل خط أنابيب الاستخراج الخاصة بك على جميع 50 وقياس المقابلة الدقيقة ودقة مستوى المجال وتوافق النموذج. حدد أسوأ الحقول للاستخراج بشكل صحيح.

5. إضافة "درجات الثقة" إلى خط أنابيب استخراجك. لكل حقل استخراج، تقدير مدى ثقة النموذج (بناء على احتمالات الرمز، أو عن طريق تشغيل استخراج 3 مرات وتقييم التماسك). علامة الحقول منخفضة الثقة للمراجعة البشرية.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| JSON mode | "Returns JSON" | API flag that guarantees syntactically valid JSON output, but does not enforce any particular schema |
| Structured output | "Typed JSON" | Output that matches a specific JSON Schema with correct keys, types, and constraints |
| Constrained decoding | "Guided generation" | At each token position, mask out tokens that would produce invalid output -- guarantees 100% schema compliance |
| JSON Schema | "A JSON template" | A declarative language for describing the structure, types, and constraints of JSON data (used by OpenAPI, JSON Forms, etc.) |
| Pydantic | "Python dataclasses+" | Python library that defines data models with type validation, used by FastAPI and Instructor to generate JSON Schemas |
| Function calling | "Tool use" | LLM outputs a structured function invocation (name + typed arguments) instead of free text -- OpenAI and Anthropic both support this |
| Instructor | "Pydantic for LLMs" | Python library that wraps LLM clients to return validated Pydantic instances, with automatic retry on validation failure |
| Token masking | "Filtering the vocabulary" | Setting specific token probabilities to zero during generation so the model cannot produce them |
| Schema compliance | "Matches the shape" | The output has every required field, correct types, values within constraints, and no extra disallowed fields |
| Retry loop | "Try again until it works" | Send validation errors back to the model and ask it to fix the output -- Instructor does this automatically, up to a configurable max |

## المزيد من القراءة

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)-- الوثائق الرسمية لـ " JSON Schema " القيود القيودية القيودية في API OpenAI
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702)-- ورقة الخطوط المخططية، وتصف كيفية تجميع مخططات JSON في آلات الحالة المحدودة للقيود على مستوى الرمز
- [Instructor documentation](https://python.useinstructor.com/)-- المكتبة القياسية للحصول على نتائج مهيكلة من أي ماجستير في العلوم مع التحقق من معايير و إعادة الاختبار
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)-- كيف ينفذ كلود الناتج المهيكلي عبر استخدام الأدوات مع JSON Schema input_schema
- [JSON Schema specification](https://json-schema.org/)-- المواصفات الكاملة لغة النظام المستخدمة من قبل كل نظام إنتاج مهيكلي رئيسي
- [Outlines library](https://github.com/outlines-dev/outlines)-- إعداد مصدر مفتوح مقيد باستخدام regex و JSON Schema مرتبة إلى آلات الحالة المحدودة
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100)-- محرك اللغة الحديث الحالي؛ مجموعة أوتوماتيكية تدفع أسفل التي تغطي الرموز عند ~ 100 ns / الرموز.
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094)-- إطار الورق LMQL القيود ك لغة استفسار مع قيود النوع والقيمة.
- [Microsoft Guidance (framework docs)](https://github.com/guidance-ai/guidance)-- التوليد المحدود القائم على العلامات التشريعية؛ إضافة معتاد على البائع إلى الخطوط المخططية و XGrammar.
