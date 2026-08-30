# MDP, Devletler, Eylemler ve Ödüller

> Markov Karar Süreci beş şeyden oluşur: durumlar, eylemler, geçişler, ödüller, indirim. RL  Q-öğrenme, PPO, DPO, GRPO 'daki her şey bu şekil üzerinde optimize eder. Bir kez öğrenin, kalan güçlendirme öğrenimini ücretsiz olarak okuyun.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## Sorun

Bir satranç robotu ya da bir stok planlayıcısı ya da bir ticaret ajanı ya da bir mantık modeli eğitilen PPO döngüsü yazıyorsunuz.

Gözetimli öğrenme size verir .`(x, y)`Bu, bir çiftin bir fonksiyona uygun olmasını ve bir fonksiyona uygun olmanızi ister. Güçlendirme öğrenimi size etiket vermez. Sadece bir devlet akımı, yaptığınız eylemler ve bir skalar ödül.

Bu akıştan, bunu resmileştirmeden öğrenemezsiniz. "Ne gördüm", "ne yaptım", "ne oldu, "ne kadar iyiydi"  her biri mantık edebileceğiniz bir nesne haline gelmelidir. Bu resmileştirme Markov Karar Süreci'dir. Bu aşamada bulunan her RL algoritması, sonunda RLHF ve GRPO döngüleri de dahil olmak üzere, bu şekil üzerinde optimize eder.

## Anlaşım

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**The five objects.**

- **States** `S`Agentin karar vermesi gereken her şey. GridWorld'da hücre, satrançta tahta, LLM'de bağlam penceresi ve herhangi bir hafıza.
- **Actions** `A`Seçenekler, yukarı/ aşağı/ sola/ sağa hareket et.
- **Transitions** `P(s' | s, a)`- Durum verildiğinde .`s`ve eylemler`a`Satrançta belirgin, stokastik envanterde, LLM'de neredeyse belirgin.
- **Rewards** `R(s, a, s')`- Skalaar sinyal. Kazanç = +1, Kayıp = -1. Gelir eksi maliyet.
- **Discount** `γ ∈ [0, 1)`Gelecekteki ödülün ne kadarı var ?`γ = 0.99`~ 100 adımlık ufuk alır; `γ = 0.9`- 10'u alır.

**The Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`Gelecek sadece mevcut durumdan bağlıdır. Eğer öyle değilse, devlet temsilcisi eksiktir.

**Policies and returns.**Bir politika`π(a | s)`Haritalar, hareket dağılımlarına ilişkin durumları gösterir.`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`gelecek ödüllerin indirimli toplamıdır.`V^π(s) = E[G_t | s_t = s]``s`Politikası altında`π`- Q değerini .`Q^π(s, a) = E[G_t | s_t = s, a_t = a]`Bu iki işlemden birini tahmin eden her RL algoritması, daha sonra iyileştirir.`π`Bu yüzden.

**The Bellman equations.**Bu aşamada her şeyin kullandığı sabit nokta denklemleri:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Bu bölünme beklenen "bu adım ödül" artı "hiç yere düştüğünüz yerin indirim değeri" olarak geri döner. Tekrarlı. 9. aşamada her algoritma bu denklemi ya bir yaklaşım (dinamik programlama), ondan örnekler (Monte Carlo) veya bir adım (zaman farkı) ile yeniden başlatır.

```figure
discount-horizon
```

## Yapın

### Adım 1: Küçük bir deterministik MDP

Agent yukarıdan sola, terminal aşağıdan sağ, ödül adım başına -1'dir, eylemler.`{up, down, left, right}`Bakın .`code/main.py`- Evet .

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

Beş çizgi, tüm çevre bu. Deterministik geçişler, sürekli adım cezaları, terminal durumu absorbe etmek.

### İkinci adım: Bir politika oluşturmak

Bir politika, devletten eylemlere dağılım fonksiyonudur. En basit: eşsiz rastgele.

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

Bu 4×4 tablo için ortalama geri dönüş -60 ila -80 civarındadır. Optimal geri dönüş -6 (sürekli çizgi yolu aşağı doğru) dir. Bu boşluğu kapatmak 9. aşamada her şey.

### Adım 3: hesaplama`V^π`Tam olarak Bellman denkleminden

Küçük MDP'ler için Bellman denkleminin bir çizgisi vardır. Sayım durumları, beklentiyi uygulayın, değerlerin değişmesini durdurana kadar tekrarlayın.

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

Bu, Sutton & Barto'daki ilk algoritmadır ve ardından gelen her RL yöntemi için teorik temeldir.

### Dördüncü adım:`γ`fiziksel anlamı olan bir hiperparametre

Etkili ufuk yaklaşık olarak `1 / (1 - γ)`- Evet .`γ = 0.9`→ 10 adım. `γ = 0.99`→ 100 adım. `γ = 0.999`→ 1000 adım.

Çok düşük ve ajan kısa görüşlü davranır. Çok yüksek ve kredi tahsis gürültülü hale gelir, çünkü birçok erken adım uzun vadeli ödül için sorumluluk paylaşır. LLM RLHF tipik olarak kullanır `γ = 1`Çünkü bölümler kısa ve sınırlı.`0.95–0.99`Uzun üfüre strateji oyunları kullanıyor`0.999`- Evet .

## Tuzaklar

- **Non-Markovian state.**Eğer son üç gözlemden karar vermek istiyorsanız, "devlet" sadece mevcut gözlem değildir.
- **Sparse rewards.**Sadece kazançlı ödüller, büyük devlet alanlarında öğrenmeyi neredeyse imkansız hale getirir.
- **Reward hacking.**Bir vekil ödülünü optimize etmek genellikle patolojik bir davranış üretir. OpenAI'nin tekne yarışları ajanı yarışı bitirmek yerine sonsuza dek güç toplayarak çevrelerde döner.
- **Discount mis-spec.** `γ = 1`Sonsuz ufuklı bir görevin her değerini sonsuz yapar.`γ < 1`- Evet .
- **Reward scale.**{+100, -100} vs {+1, -1} ödülleri aynı optimum politikaları verir ama büyük ölçüde farklı gradient büyüklükleri.`[-1, 1]`-PPO/DQN'e bağlanmadan önce.

## Kullan

2026 yığın, her RL borusunu kodla temas etmeden önce bir MDP'ye düşürür:

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

Bir eğitim döngüsünü yazmadan önce beş tupleyi yazın. "RL çalışmıyor" hata raporlarının çoğu kağıt üzerinde kırılmış bir MDP formülasyonuna kadar uzanır.

## Gönder

- Kaydet .`outputs/skill-mdp-modeler.md`- ...

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

## Egzersizler

1. **Easy.**4×4 GridWorld ve rastgele politika uygulaması uygulamak `code/main.py`- 10.000 bölüm çalıştırın. Geri dönüş ortalaması ve STD rapor edin.
2. **Medium.**Çık .`policy_evaluation`- Evet .`γ ∈ {0.5, 0.9, 0.99}`- Üniformal rastgelelik politikası için.`V`Her biri için 4×4 şebekesi olarak. Terminal yakınındaki durum değerlerinin neden daha büyük olan daha hızlı büyüdüğünü açıklayın.`γ`- Evet .
3. **Hard.**GridWorld ' i stohastik çevir: her eylem olasılık ile bitişik bir yöne kayar `p = 0.1`- Üniforma politikasını yeniden değerlendirmek.`V[start]`- Daha iyi mi kötü mi?

## Anahtar Terimler

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

## Daha Fazla Okumak

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf)3. bölüm MDP'leri ve Bellman denklemlerini kapsar; 1. bölüm her sonraki derslerin altında bulunan ödül hipotezini motive eder.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) Bellman denkleminin kökeni.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) derin bir RL açısından kısa bir MDP primer.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) MDP'ler ve tam çözüm yöntemleri ile ilgili operasyon-kağıt araştırma referansı.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) dinamik programlama uzmanlığı olarak MDP'lerin en temiz türü.
