# المشاريع المتعددة، الدول، الإجراءات والمكافآت

> عملية قرار ماركوف هي خمسة أشياء: الحالات، والإجراءات، والانتقالات، والمكافآت، والخصم. كل شيء في RL  Q-تعلم، PPO، DPO، GRPO  تحسن على هذا الشكل. تعلمها مرة واحدة، قراءة بقية التعلم التوطيد مجانا.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## المشكلة

أنت تكتب روبوت شطرنج أو مخطط مخزون أو وكيل تجاري أو حلقة PPO التي تدرب نموذج التفكير أربعة مجالات مختلفة، حقيقة مفاجئة واحدة: جميعها تنهي إلى نفس الكائن الرياضي.

التعلم المشرف يعطيك`(x, y)`يطلب منك التكيف مع وظيفة. تعلم التعزيز لا يعطيك علامات  فقط تدفق من الحالات، والإجراءات التي اتخذتها، ومكافأة واسعة. هل الفوز في اللعبة؟ هل قرار إعادة التعبئة توفير المال؟ هل التجارة تحقق ربحا؟ هل الوهم الذي أنتجته للتو أدى إلى مكافأة أعلى من القاضي؟

لا يمكنك التعلم من هذا التيار حتى تقوم بتسهيله. "ما رأيته" "ما فعلته" "ما حدث بعد ذلك" "كم كان ذلك جيداً"  كل شيء يجب أن يصبح كائنًا يمكنك التفكير فيه. هذه التسهيله هي عملية قرار ماركوف. كل خوارزمية RL في هذه المرحلة، بما في ذلك حلقات RLHF و GRPO في النهاية، تحسن على هذا الشكل.

## المفهوم

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**The five objects.**

- **States** `S`كل ما يحتاج الوكيل إلى القرار به في "جريد ورلد" الخلية في الشطرنج اللوحة في ماجستير في العلوم، نافذة السياق بالإضافة إلى أي ذاكرة
- **Actions** `A`الخيارات، تحرك للأعلى/إلى الأسفل/اليسار/اليمين، تلعب حركة، إصدر رمزاً
- **Transitions** `P(s' | s, a)`. في ظل الحالة`s`والعمل`a`التمييز في الشطرنج، الاستوكاستيك في المخزون، شبه التمييز في فك التمييز
- **Rewards** `R(s, a, s')`الإشارة المتعددة الربح = +1، الخسارة = -1. الإيرادات ناقص التكلفة. معدل احتمالات التسجيل في GRPO.
- **Discount** `γ ∈ [0, 1)`كم من المكافأة المستقبلية تُعادل مع الحاضر`γ = 0.99`يشتري أفقًا من ~ 100 خطوة ؛`γ = 0.9`يشتري 10

**The Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. المستقبل يعتمد فقط على الحالة الحالية. إذا لم يكن كذلك، فإن تمثيل الدولة غير كامل

**Policies and returns.**سياسة`π(a | s)`الخرائط تعطي التوزيعات العمل.`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`هو المبلغ المخصوم من المكافآت المستقبلية.`V^π(s) = E[G_t | s_t = s]`هو العائد المتوقع بدءا من `s`في سياسة`π`. قيمة Q`Q^π(s, a) = E[G_t | s_t = s, a_t = a]`هو العائد المتوقع الذي يبدأ بعمل معين. كل خوارزمية RL تقدّر واحدة من هذين الاثنين، ثم تحسن `π`وفقًا لذلك

**The Bellman equations.**المعادلات الثابتة التي تستخدمها كل شيء في هذه المرحلة:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

هذه الانقسام المتوقعة تعود إلى "مكافأة هذه الخطوة" زائد "قيمة خصم من حيث الهبوط". التكراري. كل خوارزمية في المرحلة 9 إما تكرر هذه المعادلة إلى التقارب (برمجة ديناميكية) ، أو عينات منها (مونتي كارلو) ، أو تشغيلها خطوة واحدة (فرق الزمن).

```figure
discount-horizon
```

## بناءها

### الخطوة الأولى: MDP الحدّيّة الصغيرة

عميل يبدأ من أعلى اليسار، المحطة في الأسفل اليمين، مكافأة -1 لكل خطوة، الإجراءات`{up, down, left, right}`- انظر`code/main.py`. . .

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

خمسة خطوط، هذا هو البيئة بأكملها، عمليات انتقالٍ تحديدية، عقوبة خطوة ثابتة، امتصاص حالة نهائية.

### الخطوة الثانية: وضع سياسة

السياسة هي وظيفة من حالة إلى عملية التوزيع. أبسطها:

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

قم بتشغيل السياسة العشوائية 1000 مرة. متوسط العائد هو حوالي -60 إلى -80 لهذا اللوح 4 × 4. العائد الأمثل هو -6 (مسار الخط المستقيم إلى اليمين). إغلاق هذه الفجوة هو كل شيء في المرحلة 9.

### الخطوة الثالثة: الحساب`V^π`بالضبط عن طريق معادلة بيلمان

بالنسبة لمعدلات MDP الصغيرة هي معادلة بيلمان هي نظام خطي. قم بتحديد الحالات وتطبيق التوقعات وتكرارها حتى تتوقف القيم عن التغير.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

هذا هو تقييم السياسة المتكرر. إنه أول خوارزمية في سوتون و بارتو والأساس النظري لكل طريقة RL التي تليها.

### الخطوة الرابعة:`γ`هو مُعدل فائق معنى جسدي

الأفق الفعلي تقريباً`1 / (1 - γ)`. .`γ = 0.9`→ 10 خطوات. `γ = 0.99`→ 100 خطوة. `γ = 0.999`1000 خطوة

منخفض جداً ويتصرف الوكيل بعمق قصير. مرتفع جداً و يصبح تعيين الائتمان ضجيجاً، لأن العديد من الخطوات المبكرة تتشارك المسؤولية عن مكافأة بعيدة المستقبل.`γ = 1`لأن الحلقات قصيرة ومحدودة.`0.95–0.99`ألعاب استراتيجية طويلة الأفق تستخدم`0.999`. . .

## الفخاخ

- **Non-Markovian state.**إذا كنت بحاجة إلى الملاحظات الثلاثة الأخيرة لتحديد، فإن "الحالة" ليست مجرد الملاحظة الحالية.
- **Sparse rewards.**مكافآت النصر فقط تجعل التعلم مستحيلا تقريبا في المساحات الكبيرة. مكافآت الشكل (إشارة متوسطة) أو إطلاق مع التمثيل (المرحلة 9 · 09).
- **Reward hacking.**تحسين مكافأة الوكيل غالبا ما ينتج سلوكًا مريضًا. يدور وكيل سباق القوارب في OpenAI في دوائر يجمع القوى إلى الأبد بدلاً من إنهاء السباق. حدد دائمًا مكافأة من النتيجة المستهدفة ، وليس الوكيل.
- **Discount mis-spec.** `γ = 1`في مهمة ذات الأفق المحدود يجعل كل قيمة لا نهاية لها. دائما القمة مع إما الأفق المحدود أو`γ < 1`. . .
- **Reward scale.**مكافآت {+100، -100} مقابل {+1، -1} تعطى نفس السياسات المثلى ولكن اختلاف كبير جدا في حجم التراجع.`[-1, 1]`- قبل التوصيل إلى PPO / DQN.

## استخدمها

كومة 2026 تقلل كل خط أنابيب RL إلى MDP قبل لمس الرمز:

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

اكتب الخمسة أجزاء قبل كتابة أي حلقة تدريبية. تعود معظم تقارير الأخطاء "RL لا تعمل" إلى صيغة MDP التي تم كسرها على الورق.

## أرسله

إبقوا`outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## التمارين

1. **Easy.**تنفيذ 4 × 4 GridWorld والتنفيذ في السياسات العشوائية`code/main.py`. إشغال 10 آلاف حلقة . تقرير متوسط و ستد من العائد . مقارنة مع العائد المثالي ( - 6).
2. **Medium.**أركض`policy_evaluation`مع`γ ∈ {0.5, 0.9, 0.99}`لسياسة الاختيار العشوائي الموحد`V`كشبكة 4 × 4 لكل واحد. شرح لماذا قيم الحالة بالقرب من المحطة تزداد بسرعة مع أكبر `γ`. . .
3. **Hard.**قم بتحويل شبكة العالم إلى مستوى مستقر: كل عمل ينزلق إلى اتجاه مجاور مع احتمال `p = 0.1`إعادة تقييم سياسة المزيّة`V[start]`هل تتحسن أم تتفاقم؟

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MDP | "Reinforcement learning setup" | Tuple `(S, A, P, R, γ)` satisfying the Markov property. |
| State | "What the agent sees" | Sufficient statistic for future dynamics under the chosen policy class. |
| Policy | "Agent's behavior" | Conditional distribution `π(a \| s)` or deterministic map `s → a`. |
| Return | "Total reward" | Discounted sum `Σ γ^t r_t` from the current step. |
| Value | "How good a state is" | Expected return under `π` starting from `s`. |
| Q-value | "How good an action is" | Expected return under `π` starting from `s` with first action `a`. |
| Bellman equation | "Dynamic programming recursion" | Fixed-point decomposition of value / Q into one-step reward plus discounted successor value. |
| Discount `γ` | "Future vs present" | Geometric weight on far-future reward; effective horizon `~1/(1-γ)`. |

## المزيد من القراءة

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) الكتاب الدراسي. يغطي الفصل 3 المعادلات MDPs و Bellman؛ الفصل 1 يحفز فرضية الجائزة التي تستند إلى كل درس لاحقا.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) أصل معادلة بيلمان
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) مُبادرة MDP مُختصرة من زاوية عميقة لـ RL.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) إشارة البحث العمليات حول MDPs وأساليب الحل الدقيقة.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) أسهل استنتاج من MDPs كتميز في البرمجة الديناميكية.
