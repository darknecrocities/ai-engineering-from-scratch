# تحسين السياسة القريبة (PPO)

> يرمي A2C كل عملية تنفيذ بعد تحديث واحد. يلف PPO تراجع السياسة في نسبة أهمية خفضة حتى تتمكن من القيام بأكثر من 10 حقائق على نفس البيانات دون انفجار السياسة. Schulman et al. (2017). لا يزال الخوارزمية الافتراضية للسياسة-الترقي في عام 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## المشكلة

A2C (دروس 07) على السياسة: التراجع `E_{π_θ}[A · ∇ log π_θ]`يتطلب البيانات التي تم أخذها من *التيار* `π_θ`خذ تحديث واحد، و`π_θ`تغيرات، البيانات التي استخدمتها غير قانونية الآن إعادة استخدامها و تغير التحديد

التشغيل مكلف. على Atari، التشغيل واحد عبر 8 envs × 128 خطوات = 1024 انتقالات وعشرة ثوان من الوقت البيئي. إلقاء ذلك بعيدا بعد خطوة تراجع واحدة هو مضيعة.

كان تحسين سياسة منطقة الثقة (TRPO ، Schulman 2015) هو التحدي الأول: تقييد كل تحديث بحيث يبقى اختلاف KL بين السياسة القديمة والجديدة أقل `δ`نظرياً نظيفة، لكن تتطلب حلًا متضامنًا لكل تحديث. لا أحد يعمل على TRPO في عام 2026.

تمت إعادة تشكيل المعلومات في الموقع، حيث تمت إعادة تشكيل المعلومات في الموقع، حيث تمت إعادة تشكيل المعلومات في الموقع، حيث تمت إعادة تشكيل المعلومات في الموقع.

## المفهوم

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**The importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

هذا هو نسبة احتمالية السياسة الجديدة مقابل السياسة التي جمعت البيانات. `r_t = 1`لا يعني أي تغيير`r_t = 2`يعني أن السياسة الجديدة أكثر عرضة للانتقال`a_t`مثل القديمة

**The clipped surrogate.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

شروطين:

- إذا كانت الميزة`A_t > 0`و يحاول النسبة أن تنمو`1 + ε`، المقطوعة تسطح التراجع  لا تدفع عمل جيد أبعد من `+ε`فوق الاحتمالات القديمة
- إذا كانت الميزة`A_t < 0`و يحاول النسبة أن تنمو`1 - ε`(ما يعني أننا سنجعل خطوة سيئة أكثر احتمالا مقارنة مع تقليصها المقطوعة) ، كليب كابس التراجع  لا يدفع خطوة سيئة أسفل `-ε`. . .

- نعم`min`يتعامل الاتجاه الآخر: إذا تحرك النسبة في الاتجاه * المفيد * ، فإنك لا تزال تحصل على التراجع (لا وجود قطع على الجانب الذي سيؤذيك).

نموذجي`ε = 0.2`. رسم الهدف كعمل من`r_t`: وظيفة خطية على شكل قطعة مع سقف مسطح على الجانب "الجيد" والطابق مسطح على الجانب "سيئ".

**The full PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

نفس الهيكل الممثل-النقدي مثل A2C. ثلاثة معايير، عادة `c_v = 0.5`،`c_e = 0.01`،`ε = 0.2`. . .

**The training loop.**

1. جمع`N × T`الانتقالات عبر `N`بيئات متوازية`T`كل خطوة
2. احسب المزايا (GAE) ، وتجمدها كمتواصلات.
3. تجميد`π_{θ_old}`كقطة من التيار`π_θ`. . .
4. لأجل`K`فترة، لكل مجموعة صغيرة من`(s, a, A, V_target, log π_old(a|s))`:
   - الحساب`r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`. . .
   - التطبيق`L^{CLIP}`+ فقدان القيمة + إنتروبي
   - خطوة تدريجية
5. إرمي التنفيذ، عودي إلى الخطوة الأولى

`K = 10`وبالطائفة الصغيرة من 64 هو مجموعة متطابقة من المعايير.

**KL-penalty variant.**اقترحت الورقة الأصلية بديلاً باستخدام عقوبة KL قابلة للتكيف: `L = L^{PG} - β · KL(π_θ || π_old)`مع`β`تم تعديلها بناءً على KL الملاحظ. أصبح نسخة القصص المهيمنة؛ والفرع KL يبقى في RLHF (حيث يكون KL إلى سياسة المرجعية قيودا منفصلة تريد دائما على أي حال).

```figure
ppo-clip
```

## بناءها

### الخطوة الأولى: التقاط`log π_old(a | s)`في وقت الإطلاق

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

يتم التقاط اللقطة مرة واحدة، في وقت الإطلاق. لا تتغير خلال فترات التحديث.

### الخطوة الثانية: حساب ميزات GAE (المدرسة 07)

نفس A2C، تطبيع على جميع أنحاء اللحظة.

### الخطوة الثالثة: تحديثات بديلة مقطوعة

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

نمط "التراجع إلى الصفر" هو قلب PPO. إذا تمت التحويل السياسة الجديدة بالفعل إلى اتجاه مفيد، فإن التحديث يتوقف.

### الخطوة الرابعة: القيمة والإنتروبي

إضافة MSE القياسية إلى الهدف النقدي ومكافأة الإنتروبي على الفاعل، نفس A2C.

### الخطوة 5: التشخيص

ثلاثة أشياء يجب مشاهدتها في كل تحديث:

- **Mean KL** `E[log π_old - log π_θ]`يجب أن تبقى في`[0, 0.02]`إذا تمرّ`0.1`، تقليل`K_EPOCHS`أو`LR`. . .
- **Clip fraction** الجزء من العينات التي يقع نسبةها خارجها `[1-ε, 1+ε]`يجب أن يكون`~0.1-0.3`إذا`~0`، المقطع لا يطلق أبدا → رفع `LR`أو`K_EPOCHS`إذا`~0.5+`، أنت تتجاوز حدة التنفيذ → خفضهم.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. متريكه النقدي الجوده يجب أن يرتفع نحو 1 كما يتعلم النقدي

## الفخاخ

- **Clip coefficient mistuned.** `ε = 0.2`هو المعيار الفعلي.`0.1`يجعل التحديثات خجولة جداً`0.3+`يدعو لعدم الاستقرار
- **Too many epochs.** `K > 20`يزعج بشكل روتيني لأن السياسة تتحرك بعيداً عن`π_old`. فترات الحد الأقصى، خاصة للشبكات الكبيرة
- **No reward normalization.**مقياس مكافأة كبيرة تستهلك نطاق المقاطع. عادي مكافآت (تشغيل std) قبل ميزات الحوسبة.
- **Forgetting advantage normalization.**تعاديل المعدل الصفر لكل مجموعة وحدة ستد هو معيار. تخطي ذلك يدمّر PPO على معظم المعايير.
- **Learning rate not decayed.**يُستفيد PPO من تدهور LR الخطي إلى الصفر. غالباً ما يكون LR الثابت أسوأ.
- **Importance ratio math errors.**دائماً`exp(log_new - log_old)`للاستقرار الرقمي، لا `new / old`. . .
- **Wrong gradient sign.**أقصى قدر من الاختيارات`-L^{CLIP}`علامة مُعكسة هي أخطاء (بي.بي.او) الأكثر شيوعاً

## استخدمها

إن PPO هو خوارزمية RL الافتراضية لعام 2026 عبر عدد مفاجئ من المجالات:

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (Lesson 12) — PPO variant without critic |
| Preference-only data | DPO — closed-form collapsing of PPO+KL, no online sampling |

تشكل PPO * الخسارة *  المقطع بديل + القيمة + الإنتروبي  هو الرفوف ل DPO ، GRPO ، وكل خط أنابيب RLHF تقريبًا.

## أرسله

إبقوا`outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## التمارين

1. **Easy.**إشغال PPO على 4 × 4 GridWorld مع `ε=0.2, K=4`- مقارنة كفاءة العينة مع A2C (حلقة واحدة لكل عملية التنفيذ) في مراحل البيئة المقابلة.
2. **Medium.**تفتيش`K ∈ {1, 4, 10, 30}`. عودة المسار مقابل خطوات البيئة وتتبع متوسط KL لكل تحديث.`K`هل ستنفجر (كيل) في هذه المهمة؟
3. **Hard.**استبدل الاختراق المستبدل مع عقوبة KL تكييفية (`β`مضاعفة إذا`KL > 2·target`، نصف إذا `KL < target/2`) مقارنة العائد النهائي، الاستقرار، وبدون كليب.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; deviation from the policy that collected the data. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; flat gradient past the clip on beneficial side. |
| Trust region | "TRPO / PPO intent" | Limit each update's KL to guarantee monotone improvement. |
| KL penalty | "Soft trust region" | Alternative PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptive `β`. |
| Clip fraction | "How often clipping triggers" | Diagnostic — should be 0.1-0.3; outside means mistuned. |
| Multi-epoch training | "Data reuse" | K epochs on each rollout; variance cost traded for sample efficiency. |
| On-policy-ish | "Mostly on-policy" | PPO is nominally on-policy but K>1 epochs uses slightly-off-policy data safely. |
| PPO-KL | "The other PPO" | KL-penalty variant; used in RLHF where KL-to-reference is already a constraint. |

## المزيد من القراءة

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)-الورقة
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)(تريبو) ، سلف (بيبو)
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) كل مفاتيح PPO متزايدة تم إزالة.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) InstructGPT: وصفة PPO-in-RLHF
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) نظيف المعرض الحديث مع PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) إشارة PPO ملف واحد يستخدم من قبل العديد من الأوراق.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) وصفة الإنتاج لـ PPO على نماذج اللغة؛ اقرأ جنبا إلى جنب مع الدروس 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) ورقة "37 تحسينات مستوى الرمز" ؛ أي خدوش PPO تحمل الحمل والتي هي شعبية.
