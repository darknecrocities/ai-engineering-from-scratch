# 快速缓存和语义缓存经济

> **Pricing snapshot dated 2026-04.**下面的数字索赔反映了本课程发布时捕获的供应商利率卡;在下游报价之前,请与链接的文件进行验证.

> 缓存发生在两个层次.L2 (提供商级) 提示/预写缓存重复使用重复预写的注意 KV  人类的提示缓存文件在长时间提示上宣传到90%的成本降低和85%的延迟降低;对于Claude 3.5 Sonnet缓存读取是$0.30/M vs $清新3.00/M,为5分钟的TTL和1小时的TTL选项的2倍写费 (docs.anthropic.com, 2026-04). 开启AI提示缓存自动适用于提示 ≥1024个代币和价格缓存输入大约90%折扣对新鲜 (platform.openai.com, 2026-04);每个模型的确切缓存率取决于现场率卡. 应用程序级语义缓存将完全跳过LLM在嵌入类似性击中. 供应商"95%准确性"指匹配准确性,而不是击率 报告的生产击率从10% (开放式聊天) 到70% (结构化常见问题);任何供应商都没有发布官方基准,所以把这些视为社区远程测量而不是保证. 生产陷:并行化杀死缓存 (在第一个缓存写之前发出的N并行请求可以膨胀支出多倍),前内部的动态内容完全防止缓存击中. 项目发现报告通过从可缓存的前中移动动动态文本,从7%到74%的击中率 (2025-11)

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 区分L2提示/预写缓存 (在提供商中重复使用KV) 与L1语义缓存 (在类似提示时绕过LLM).
- 解释人类的故事`cache_control`显而易见的标记和两个TTL选项 (5分钟对比1小时) 及其价格乘法.
- 计算预期的月性节省,以击中率,快速/响应混合和代币价格.
- 给出一个反对比模式,使账单膨胀5到10倍,以及一个反对动态内容的反模式,

## 问题

您将快速缓存添加到您的RAG服务中.账单保持平稳.您测量了击率;它是7%.您的提示看起来是静态,但它们不是.系统提示包括当前的日期格式为分钟,请求 ID,和随机的例子重新排序为多样性.每个请求写出一个新的缓存输入,读取零.

您的代理人每次用户问话,每次运行10次并行工具调用.前10次预存写完成之前,所有10次都到达提供商.10次写,零次读.您的账单是5-10倍"预存"所需的成本.

缓存是协议,不是旗.

## 概念

### L2 提供商提示/预设缓存

提供商将注意力KV存储为可缓存的预写,并在下一次与预写匹配的请求上再使用它.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**具体情况`cache_control`标记在请求中.您标记哪些块可以缓存. TTL: 5 分钟 (写费 1.25x 基础) 或 1 小时 (写费 2x 基础).缓存读取: $0.30/M on Claude 3.5 Sonnet vs $价格因车型而异 (Opus/Haiku 单独发布); 总是通过直播价格页面进行交叉检查.

**OpenAI**现有gpt-4o/gpt-5率卡上,缓存输入比新增的约10倍便宜.文件和发布说明都没有公布官方的关键率基线;社区报告在3060%左右的细致的提示设计下集成.监测`usage.cached_tokens`为了测量自己的.

**Google (Gemini)**通过明确的API进行内存存; 1M-代币内存意味着内存存付出更多.

**Self-hosted (vLLM, SGLang)**: 17 · 06 阶段涵盖RadixAttention 您自己的计算模式.

### L1 应用级语义缓存

在打电话给LLM之前,按点击提示,嵌入它,并寻找类似的缓存请求 (值以上的类似性,通常是0.95+).在击中,返回缓存响应.在错误时,打电话给LLM并缓存结果.

开源:Redis向量类似性,GPTCache,Qdrant.商业:Portkey Cache,Helicone Cache.

供应商的准确性要求是指返回缓存响应的含义性适当度,而不是你打的频率.

- 开放式聊天: 10-15%.
- 结构性常见问题/支持:40-70%.
- 代码问题:20-30% (小变量杀死击中).
- 语音代理重复提示:50-80% (语音正常化固定设置).

### 平式化反模式

您的代理人同时进行10次工具调用.所有10次都具有相同的4K代码系统提示.人类缓存写作是按要求进行的;提供者看到提示后,第一个缓存写作完成了大约300ms. 2-10次请求都在同一毫秒窗口中到达,每个缓存都会出现错误.您支付10次写费,0次阅读折扣.

修复: 随序列-第一  单独请求 1,然后在 1' 缓存填充后,启动 2-10 秒. 添加300 ms到第一个工具调用;节省 5-10 倍的账单.

### 动态内容反模式

你的系统提示看起来像:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是独一无二的,每一个请求都写着,零打.

修复:将真正静态的内容移动到可缓存的前置;在缓存边界后添加动态内容:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

通过此方式,ProjectDiscovery从7%上升到74%的缓存击中率,并发布了解剖学.

### 堆批量+夜间工作负载的缓存

批量API (阶段17 · 15) 在24小时转换时提供50%的折扣. 存储输入上方为您提供了10倍的额外. 通过堆叠,一夜间分类,标签和报告生成工作负载可以降至同步未加载成本的10%

### 你应该记住的数字

价格点从链接的供应商文件中被捕获2026-04年,每几个月都会被转移到重新检查之前依赖它们.

- 克劳德3.5索尼特的缓存读数:0.30美元/万,比新输入的价格约是10倍.
- 人类缓存写入溢价: 1.25x (5分钟TL) 或 2x (1小时TL).
- 开AI自动缓存:适用于提示 ≥1024个代币;缓存输入价格约为当前价格卡的新输入的10% (platform.openai.com).
- 语义缓存击中率 (社区报告): ~10%开放聊天;高达 ~70%结构化的FAQ. 不是供应商记录的基线.
- 项目发现: 7% → 74% 的击中率通过从前移动动态 (项目博客, 2025-11).
- 平式化反模式:在N平行请求错过第一个缓存写时,典型报告510x账单通胀.

```figure
semantic-cache-hit
```

## 用它

`code/main.py`报告中显示了率,账单,并显示了并行处罚.

## 运送它

这一课产生了`outputs/skill-cache-auditor.md`鉴于快速的模板和流量,审计可存储性,并建议进行重组.

## 运动

1. 跑步`code/main.py`换并行标志. 账单有多少变化?
2. 系统提示有日期,请移动,显示前后的按率.
3. 根据您的请求到达率,计算1小时的TTL (2x写) 与5分钟的TTL (1.25x写) 的破解平衡.
4. 在0.95的门时,语义缓存达到20%.在0.85时,它达到50%但你看到错误的缓存响应.
5. 按用户问题进行10个并行子查询. 为了缓存友好性,再写,而不需要添加端到端延迟.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

## 进一步阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)官方`cache_control`语义和TL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)自动缓存行为和资格.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
