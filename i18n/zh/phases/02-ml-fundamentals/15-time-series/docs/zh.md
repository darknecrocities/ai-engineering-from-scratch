# 时间系列的基本知识

> 如果先检查停滞,过去的表现确实能预测未来的结果.

**Type:** Build
**Language:**字符串
**Prerequisites:** Phase 2, Lessons 01-09
**Time:** ~90 minutes

## 学习目标

- 分解时间序列成趋势,季节性和残留组件,测试静止性
- 实现延迟功能和滚动统计数据,将时间序列转化为监督学习问题
- 建立一个前进验证框架,以防止未来数据泄露到培训中
- 解释为什么随机列车/测试分区对时间序列不有效,并证明性能差距与适当的时间分区

## 问题

您有时间排列的数据,每天的销售,每小时的温度,每分钟的CPU使用,每周的股价.

你需要一个标准的 ML 工具包:随机列车/测试分区,交叉验证,功能矩阵进入,预测出.

时间系列打破了标准ML依赖的假设. 样本并不独立 - - 今天的温度取决于昨天的温度. 随机分区泄漏未来的信息到过去. 后测试中看起来很好的功能在生产中失败,因为它们依赖于随时间而改变的模式.

随机交叉验证的模型可以获得95%的准确性,但如果进行适当的时间评估,则可能会获得55%.

这一课涵盖了基本知识:什么使时间数据变得不同,如何诚实地评估模型,以及如何将时间系列转化为标准ML模型可以使用的功能.

## 概念

### 时间系列是什么不同?

标准ML假设i.i.d. - 独立且分布相同.每个样本都是从相同的分布中抽取的,独立于其他样本.时间系列违反了以下两种:

- **Not independent.**今天的股价取决于昨天的股价. 本周的销售与上周的销售相对.
- **Not identically distributed.**随着时间的推移,分销量变化.

这些违规行为并不小,而是改变了你构建功能的方式,

```mermaid
flowchart LR
    subgraph IID["Standard ML (i.i.d.)"]
        direction TB
        S1[Sample 1] ~~~ S2[Sample 2]
        S2 ~~~ S3[Sample 3]
    end
    subgraph TS["Time Series (not i.i.d.)"]
        direction LR
        T1[t=1] --> T2[t=2]
        T2 --> T3[t=3]
        T3 --> T4[t=4]
    end

    style S1 fill:#dfd
    style S2 fill:#dfd
    style S3 fill:#dfd
    style T1 fill:#ffd
    style T2 fill:#ffd
    style T3 fill:#ffd
    style T4 fill:#ffd
```

在标准ML中,样本是可替换的. 混动它们不会改变任何东西. 在时间序列中,顺序是一切. 混动破坏信号.

### 时间系列的组成部分

每个时间系列都是:

```mermaid
flowchart TD
    A[Observed Time Series] --> B[Trend]
    A --> C[Seasonality]
    A --> D[Residual/Noise]

    B --> E[Long-term direction: up, down, flat]
    C --> F[Repeating patterns: daily, weekly, yearly]
    D --> G[Random variation after removing trend and seasonality]
```

- **Trend**长期方向:收入每年增长10% 全球气温上升
- **Seasonality**零售销售量在12月升.空调使用量在7月升.
- **Residual**随着移除趋势和季节性之后所剩下的任何东西. 如果残留物看起来像白噪声,分解就捕获了信号.

### 固定性

时间序列是静止的,如果其统计性质 (平均值,变异性,自相相性) 随着时间的推移不变.大多数预测方法都假设静止性.

**Why it matters:**没有固定的系列有漂移的平均值.一个基于1月的数据训练的模型已经学到了与2月的平均值不同.

**How to check:**计算滚动平均值和滚动标准偏差在窗户上.如果它们漂移,则系列是非静止的.

**How to fix:**区分. 代替模拟原价值,模拟连续值之间的变化:

```
diff[t] = value[t] - value[t-1]
```

如果一个分化轮不使系列停滞,请再次应用 (二级分化).大多数现实世界系列需要最多两轮.

**Example:**

首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页: 首页:
第一个差异: [2, 4, 6, 8] (仍然向上趋势)
第二种差异: [2,2, 2] (恒定--静止)

实际上,你很少需要超过两个轮. 实际上,你很少需要两个轮.

**Formal test:**增强式式式测试 (ADF) 是静止性的标准统计测试.零假设是"系列是非静止的".0.05以下的p值意味着你可以拒绝零并得出静止性结论.我们从零开始不实现ADF (它需要无比分分布表),但我们的代码中的滚动统计方法提供了实际的视觉检查.

### 自身相关性

自动相对度测量时间 t 的值与时间 t-k 的值相对度 (过去的 k 步骤).自动相对度函数 (ACF) 为每个 lag k 绘制了这种相对度.

**ACF tells you:**
- 如果在5步后ACF下降到零,则5步前的值是无关的.
- 如果ACF在12次 (月度数据) 后期上升,则会有年度季度性.
- 运用延迟到ACF变得微不足道的地方.

**PACF (Partial Autocorrelation Function)**如果今天与3天前相对应,因为两者都与昨天相对应,PACF在3次滞后时将是零,而ACF在3次滞后时不会.

### 延迟特征:将时间系列转化为监督学习

标准ML模型需要一个特征矩阵X和一个目标y.时间系列给你一个单一的值列.桥梁是滞后特征.

设置一个后期的位置,

| lag_2 | lag_1 | target |
|-------|-------|--------|
| 10    | 12    | 14     |
| 12    | 14    | 13     |
| 14    | 13    | 15     |

任何ML模型 (线性回归,随机森林,梯度增强) 都可以从滞后预测目标.

您可以设计的其他功能:
- **Rolling statistics:**平均值,std,min,max 在最后的k值上
- **Calendar features:**周日,月,节日,周末
- **Differenced values:**与前一步的变化
- **Expanding statistics:**累计平均,累计总额
- **Ratio features:**现行值/滚动平均值 (与最近的平均值有多远)
- **Interaction features:**_1 * 周日 (周日对动力的影响)

**How many lags?**如果 ACF 值高于10个滞后,则至少使用10个滞后.如果有周季节性,则包括7个滞后 (可能14个).更多的滞后给模型更多的历史,但也增加了更大的功能,增加了过度适应的风险.

**The target alignment trap.**创建时,目标必须是时间 t 的值,所有功能必须使用时间 t-1 或更早的值.如果你意外地将时间 t 的值作为一个功能,你有一个完美的预测器 - - 和一个完全无用的模型.这是时间系列功能工程中最常见的错误.

### 经验证

这就是这个课程中最重要的概念.标准的k倍交叉验证随机分配样本进行训练和测试.对于时间序列,这泄露未来的信息.

```mermaid
flowchart TD
    subgraph WRONG["Random Split (WRONG)"]
        direction LR
        W1[Jan] --> W2[Mar]
        W2 --> W3[Feb]
        W3 --> W4[May]
        W4 --> W5[Apr]
        style W1 fill:#fdd
        style W3 fill:#fdd
        style W5 fill:#fdd
        style W2 fill:#dfd
        style W4 fill:#dfd
    end

    subgraph RIGHT["Walk-Forward (CORRECT)"]
        direction LR
        R1["Train: Jan-Mar"] --> R2["Test: Apr"]
        R3["Train: Jan-Apr"] --> R4["Test: May"]
        R5["Train: Jan-May"] --> R6["Test: Jun"]
        style R1 fill:#dfd
        style R2 fill:#fdd
        style R3 fill:#dfd
        style R4 fill:#fdd
        style R5 fill:#dfd
        style R6 fill:#fdd
    end
```

进行前进验证:
1. 运行数据到时间 t
2. 在时间 t+1 (或 t+1 到 t+k 多步) 中预测
3. 滑向前窗户
4. 复制

每个测试折叠只包含所有训练数据后的数据. 没有未来泄漏. 这让你一个诚实的估计模型将在部署时表现如何.

**Expanding window**通过使用所有历史数据进行培训 (窗口增长).**Sliding window**使用固定尺寸的训练窗口 (窗口幻灯片). 扩展使用当你认为旧数据仍然相关时. 改变世界和旧数据伤害时使用幻灯片.

### 艾里马直觉

亚里马是经典时间序列模型. 它有三个组成部分:

- **AR (Autoregressive):**预测从过去的值.AR(p) 使用最后的p值.
- **I (Integrated):**为了实现静止性,进行分化.
- **MA (Moving Average):**预测从过去的预测错误.

根据ACF/PACF分析或自动搜索 (自动ARIMA) 选择p,d,q.

我们不会从零开始实现ARIMA,它需要数字优化,这超出了本课程的范围.关键的见解是了解每个组件的作用,

### 什么时候使用

| Approach | Best For | Handles Seasonality | Handles External Features |
|----------|---------|-------------------|------------------------|
| Lag features + ML | Tabular with many external features | With calendar features | Yes |
| ARIMA | Single univariate series, short-term | SARIMA variant | No (ARIMAX for limited) |
| Exponential smoothing | Simple trend + seasonality | Yes (Holt-Winters) | No |
| Prophet | Business forecasting, holidays | Yes (Fourier terms) | Limited |
| Neural networks (LSTM, Transformer) | Long sequences, many series | Learned | Yes |

对于大多数实际问题来说,延迟功能+梯度增强是最强的起点.它自然处理外部功能,不需要静止性,并且很容易调试.

### 预测水平和战略

单步预测预测一个步骤前进.多步预测预测多步骤.有三个策略:

**Recursive (iterated):**预测一个步骤前进,使用预测作为下一步的输入. 简单,但错误积累 - 每个预测都使用了之前的预测,所以错误复合.

**Direct:**训练一个单独的模型,为每个视界.模型-1预测t+1,模型-5预测t+5.没有错误积累,但每个模型都有较少的训练样本,并且它们没有分享信息.

**Multi-output:**训练一个同时输出所有视野的模型. 通过视野共享信息,但需要支持多个输出 (或自定义输失函数) 的模型.

对于大多数实际问题,开始用短视野 (1-5 步) 的递归和长视野的直径.

### 时间序列中的常见错误

| Mistake | Why it happens | How to fix |
|---------|---------------|-----------|
| Random train/test split | Habit from standard ML | Use walk-forward or temporal split |
| Using future features | Feature at time t included by mistake | Audit every feature for temporal alignment |
| Overfitting to seasonality | Model memorizes calendar patterns | Hold out a full seasonal cycle in the test set |
| Ignoring scale changes | Revenue doubles but patterns stay | Model percentage change instead of absolute |
| Too many lag features | "More history is better" | Use ACF to determine relevant lags |
| Not differencing | "The model will figure it out" | Tree models handle trends; linear models need stationarity |

```figure
f3-series-decompose
```

## 建立它

编码在`code/time_series.py`实现了从零开始的核心构建块.

### 拉格特征创作者

```python
def make_lag_features(series, n_lags):
    n = len(series)
    X = np.full((n, n_lags), np.nan)
    for lag in range(1, n_lags + 1):
        X[lag:, lag - 1] = series[:-lag]
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], series[valid]
```

这将1D系列转换为一个特征矩阵,每个行都有最后一个`n_lags`作为特征的值,作为目标的当前值.

### 走向向的十字验证

```python
def walk_forward_split(n_samples, n_splits=5, min_train=50):
    assert min_train < n_samples, "min_train must be less than n_samples"
    step = max(1, (n_samples - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_samples)
        if train_end >= n_samples:
            break
        yield slice(0, train_end), slice(train_end, test_end)
```

每个分区都确保训练数据都在测试数据之前得到严格的准备.

### 简单的自动降低模式

纯 AR 模型只是延迟特性的线性回归:

```python
class SimpleAR:
    def __init__(self, n_lags=5):
        self.n_lags = n_lags
        self.weights = None
        self.bias = None

    def fit(self, series):
        X, y = make_lag_features(series, self.n_lags)
        # Solve via normal equations
        X_b = np.column_stack([np.ones(len(X)), X])
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.bias = theta[0]
        self.weights = theta[1:]
        return self
```

这在概念上与课02中的线性回归相同,但适用于相同变量的时间延迟版本.

### 停留性检查

该代码计算滚动统计数据以视觉和数字评估静止性:

```python
def check_stationarity(series, window=50):
    rolling_mean = np.array([
        series[max(0, i - window):i].mean()
        for i in range(1, len(series) + 1)
    ])
    rolling_std = np.array([
        series[max(0, i - window):i].std()
        for i in range(1, len(series) + 1)
    ])
    return rolling_mean, rolling_std
```

如果滚动平均波动或滚动STD变化,则系列是非静止的. 应用区分和检查再次.

代码还通过对比系列的第一半和第二半的静止性来检查. 如果介质差异超过一半的标准偏差或变异比重超过2倍,则该系列被标记为非静止.

### 自身相关性

```python
def autocorrelation(series, max_lag=20):
    n = len(series)
    mean = series.mean()
    var = series.var()
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        cov = np.mean((series[:n-k] - mean) * (series[k:] - mean))
        acf[k] = cov / var if var > 0 else 0
    return acf
```

## 用它

通过 sklearn,您可以直接使用任何回归器的延迟功能:

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_lag_features(series, n_lags=10)

for train_idx, test_idx in walk_forward_split(len(X)):
    model = Ridge(alpha=1.0)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

对于ARIMA,使用统计模型:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(5, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
```

编码在`time_series.py`通过步行前进验证来证明两种方法并进行比较.

### 时间系列 分类

 sklearn提供了`TimeSeriesSplit`执行前行验证:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

这相当于我们从零开始的`walk_forward_split`您可以使用它与`cross_val_score`其他:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
print(f"Mean score: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### 评估指标

时间系列预测使用回归指标,但在时间意识的背景下:

- **MAE (Mean Absolute Error):**平均 y_true - y_pred 语. 简单地解释在原始单位. "平均,预测是3.2度. "
- **RMSE (Root Mean Squared Error):**平均二次错误的平方根. 惩罚大错误比MAE. 使用当大错误比许多小错误更糟时.
- **MAPE (Mean Absolute Percentage Error):**平均错误/真值是100. 独立于规模,可用于不同的数组进行比较. 但在真值为零时未定义.
- **Naive baseline comparison:**总是与简单的基线进行比较.季节性天真基线预测了过去一段时间的值 (昨天,上周).如果你的模型不能击败天真,那么有什么不对.

### 滚动特征

代码显示,在7天和14天的窗口中增加滚动统计数据 (平均,STD,min,max).这些数据为模型提供了有关近期趋势和波动性的信息,而仅仅拖延特征无法捕获.

例如,如果滚动平均值上升,这表明上升趋势.如果滚动STD增加,这表明不断增长的波动性.这些是树型模型可以从中学习的模式,但线性模型不能.

## 运送它

这一课产生了:
- `outputs/prompt-time-series-advisor.md`-- 时间序列问题框架的提示
- `code/time_series.py`-- 延迟功能,前进验证,AR模型,静止性检查

### 你必须克服的基本标准

在构建任何模型之前,建立基线:

1. **Last value (persistence).**预测明天会像今天一样.
2. **Seasonal naive.**预测今天将与上周 (或去年) 的同一天相同.
3. **Moving average.**预测最后一个k值的平均值. 缓解噪音,但不能捕捉突然的变化.

如果你的精彩的ML模型失去了季节性天真的基线,你会出现错误.

### 实际的建议

1. **Start with plotting.**在进行模型之前,请绘制原始系列. 寻找趋势,季节性,异常值,结构性间断 (行为突然变化). 30 秒的视觉检查通常告诉你超过一个小时的自动分析.

2. **Difference first, model second.**如果系列有明显的趋势,在创建滞后特征之前,请区分它.树型模型可以处理趋势,但线性模型不能,区分永远不会伤害.

3. **Hold out at least one full seasonal cycle.**如果您有每周的季节性,您的测试组需要至少一个完整的星期. 如果每月,至少一个完整的月份.否则您无法评估模型是否捕获了季节性模式.

4. **Monitor in production.**随着世界变化的时间序列模型随着时间的推移而降低.随着时间的推移,随着时间的推移而降低.随着时间的推移而追踪预测错误.

5. **Beware of regime changes.**基于疫情前数据训练的模型不会预测疫情后行为. 包含已知政权变化的指标作为特征,或使用忘记旧数据的滑动窗口.

6. **Log-transform skewed series.**收入,价格和计数往往偏向右. 取日志稳定变化,使乘法模式加值,直线模型可以处理. 预测日志空间,然后指数回归原始单位.

## 运动

1. **Stationarity experiment.**通过滚动统计检查静止性,应用第一次差异化,再检查.一个四旋翼趋势需要多少次差异化?

2. **Lag selection.**根据季节系列 (周期=7) 计算ACF. 哪些滞后具有最高的自动相关性?仅使用这些滞后 (而不是连续滞后) 创建滞后特性.与使用滞后1到7相比,准确性是否提高?

3. **Walk-forward vs random split.**训练一个坡回归在滞后功能. 随机80/20分分和前进验证评估. 随机分数过度估算性能多少?

4. **Feature engineering.**加入滚动平均值 (window=7),滚动 std (window=7),以及周一功能.使用前进验证来比较这些额外的精度.

5. **Multi-step forecasting.**修改AR模型,以预测前进5步,而不是1.比较两种策略: (a) 预测一步,将预测作为下一步的输入 (反转), (b) 训练每个视界的单独模型 (直接).哪个更准确?

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Stationarity | "The stats don't change over time" | A series whose mean, variance, and autocorrelation structure are constant over time |
| Differencing | "Subtract consecutive values" | Computing y[t] - y[t-1] to remove trends and achieve stationarity |
| Autocorrelation (ACF) | "How a series correlates with itself" | The correlation between a time series and a lagged copy of itself, as a function of the lag |
| Partial autocorrelation (PACF) | "Direct correlation only" | Autocorrelation at lag k after removing the effect of all shorter lags |
| Lag features | "Past values as inputs" | Using y[t-1], y[t-2], ..., y[t-k] as features to predict y[t] |
| Walk-forward validation | "Time-respecting cross-validation" | Evaluation where training data always precedes test data chronologically |
| ARIMA | "The classic time series model" | AutoRegressive Integrated Moving Average: combines past values (AR), differencing (I), and past errors (MA) |
| Seasonality | "Repeating calendar patterns" | Regular, predictable cycles in a time series tied to calendar periods (daily, weekly, yearly) |
| Trend | "The long-term direction" | A persistent increase or decrease in the series level over time |
| Expanding window | "Use all history" | Walk-forward validation where the training set grows with each fold |
| Sliding window | "Fixed-size history" | Walk-forward validation where the training set is a fixed-length window that slides forward |

## 进一步阅读

- [Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/)时间序列预测的最佳免费教科书
- [scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)-- 斯克拉恩的前行分离器
- [statsmodels ARIMA docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)-- 诊断的ARIMA实施
- [Makridakis et al., The M5 Competition (2022)](https://www.sciencedirect.com/science/article/pii/S0169207021001874)-- 显示ML方法与统计方法的大型预测竞争
