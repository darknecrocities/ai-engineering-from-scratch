# 3D视觉 点云和NeRF

> 3D视觉有两个种类:点云是传感器的原始输出.NeRF是学习的体积场.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 1 Lesson 12 (Tensor Operations)
**Time:** ~45 minutes

## 学习目标

- 区分明确的3D表示 (点云,网格,语xel) 和隐含的3D表示 (签署距离场,NeRF)
- 了解PointNet的对称函数技巧,使神经网络的变量变异在一个无序的点组上
- 追踪NeRF前进传输:射线造,体积造,位置编码,MLP密度+颜色头
- 使用`nerfstudio`或`instant-ngp`预训练式3D重建从一组小的姿势图像

## 问题

摄像头产生2D图像.LIDAR产生一组无序的3D点.一个结构-从动作管道产生稀疏的3D键点云.NeRF从少数姿势图像中重建整个3D场景.所有这些都是"视觉",但它们都看起来没有像CNN想要的密集子.

3D视觉很重要,因为几乎每一个具有高价值的机器人任务都在3D中运行:抓住,避免障碍,导航,AR封闭,捕获3D内容.只了解2D图像的视觉工程师被锁定在最快增长的领域 (AR/VR内容,机器人,自动驾驶堆,基于NeRF的房地产或建筑的3D重建).

两个表示以不同的原因占主导地位.点云是传感器免费提供的东西.NeRF和其继任者 (3D高斯的光,神经SDF) 是你要求神经网络学习一个场景时得到的东西.

## 概念

### 点云

点云是R^3中的 N点的无序集合,可选每个点都有特征 (颜色,强度,正常).

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

两个特性使神经网络难以实现:

- **Permutation invariance**输出不得依赖点顺序.
- **Variable N**单个模型必须处理不同尺寸的云.

PointNet (Qi et al., 2017) 通过一个想法解决了这两个问题:将共享MLP应用于每个点,然后用对称函数 (最大积分组) 进行聚合.结果是一个不依赖顺序的固定尺寸向量.

```
f(P) = max_{p in P} MLP(p)
```

这就是PointNet的核心.更深层次的变体 (PointNet++,Point Transformer) 增加了层次性样本和本地聚合,但对称函数技巧没有改变.

### 点网架构

```mermaid
flowchart LR
    PTS["N points<br/>(x, y, z)"] --> MLP1["shared MLP<br/>(64, 64)"]
    MLP1 --> MLP2["shared MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["max pool<br/>(symmetric)"]
    MAX --> FEAT["global feature<br/>(1024,)"]
    FEAT --> FC["MLP classifier"]
    FC --> CLS["class logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

"共享MLP"是指每个点均独立运行相同的MLP. 实现为效率的点维度1x1 conv.

### 神经辐射场 (Neural Radiance Fields)

根据N照片的数据,我们可以重新构建一个3D场景吗?`(x, y, z, viewing_direction)`为了`(density, colour)`通过网络进行光线射线循环.

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

输出比较了染的像素与训练照片中的地面真相像素.通过染步骤,后方更新了MLP.没有3D地面真相,没有明确的几何学.

### 在 NeRF 中定位编码

尼拉的.`(x, y, z)`由于MLP偏向于低频率,因此不能代表高频率细节.NeRF通过在MLP之前将每个坐标编码为Fourier特征向量来解决这一问题:

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

转换器使用的方法是相同的,并且在扩散时间调节中再次出现 (课 10).没有它,NeRF看起来模糊.

### 量度表现

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i`传输率是多少光存活到点i.`(1 - exp(-sigma_i * delta_i))`是点i的度.`c_i`最后一个像素是沿光线的重量总数.

### 什么替代了NeRF

纯 NeRF 训练速度慢 (小时) 和染速度慢 (每张图片的秒).

- **Instant-NGP**(2022) 哈希网编码取代了MLP的位置输入;列车在秒钟内.
- **Mip-NeRF 360**处理无限场景和反化.
- **3D Gaussian Splatting**将数百万的3D高斯人取代了体积领域;列车在几分钟内,实时染.目前的生产默认.

几乎2026年每一个真正的NeRF产品都是3D高斯的光.

### 数据集和基准

- **ShapeNet** 3D CAD模型的分类和分类为点云.
- **ScanNet**实在的室内扫描,以进行分类.
- **KITTI**自动驾驶的户外LIDAR点云.
- **NeRF Synthetic**现在,**Blended MVS** 视图合成的呈现图像数据集.
- **Mip-NeRF 360**无限的真实场景.

```figure
nerf-rays
```

## 建立它

### 步骤1:PointNet分类器

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — transposed for Conv1d
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

运行在每云1024点.

### 步骤2: 位置编码

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

乘以`2^l * pi`它们的频率会逐渐提高.

### 步骤3:小 NeRF MLP

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

与原始NeRF相比较小 (具有2个深度8MLP干).足以展示建筑.

### 步骤4:沿光线进行体积成像

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) distances along the ray
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

一个光线,64个样本,复合到一个RGB像素和深度.

## 用它

为了真正的工作:

- `nerfstudio`目前的NeRF/Instant-NGP/Gaussian Splatting参考库.命令行加上网页浏览器.
- `pytorch3d`可分化染,点云工具,网页操作.
- `open3d`点云处理,注册,可视化.

对于部署,3D高斯式光器已大大取代纯 NeRF,因为它使其速度快100倍.重建质量可比较.

## 运送它

这一课产生了:

- `outputs/prompt-3d-task-router.md`基于任务和输入数据,一个提示将其引导到正确的3D表示 (点云,网格,语音,NeRF,高斯)
- `outputs/skill-point-cloud-loader.md`写PyTorch的技能`Dataset`对于 .ply / .pcd / .xyz文件,正常的标准化,中心化和点样本.

## 运动

1. **(Easy)**显示PointNet是变量不变的:运行相同的云两次,一次是混合点. 检查输出均等到浮点噪音.
2. **(Medium)**实现最小射线生成函数,鉴于相机内在性和姿势,为每一个H x W图像的像素产生射线起源和方向.
3. **(Hard)**训练一个TinyNeRF在合成数据集上呈现色立方体的视图 (通过可分化呈现或简单的射线追踪器生成).报告在1,10和100期的呈现损失.该模型在哪个时代产生可识别的视图?

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Point cloud | "3D points from LIDAR" | Unordered set of (x, y, z) + optional features per point |
| PointNet | "First neural net on point clouds" | Shared MLP per point + symmetric (max) pool; permutation-invariant by construction |
| NeRF | "MLP that is the scene" | Network mapping (x, y, z, dir) to (density, colour); rendered by ray casting |
| Positional encoding | "Fourier features" | Encode each coordinate into sin/cos at multiple frequencies to overcome MLP low-frequency bias |
| Volumetric rendering | "Ray integration" | Composite samples along a ray into a single pixel using transmittance and alpha |
| Instant-NGP | "Hash-grid NeRF" | Replaces NeRF's coordinate MLP with a multi-resolution hash grid; 100-1000x faster |
| 3D Gaussian splatting | "Millions of Gaussians" | Scene = collection of 3D Gaussians; renders in real time, trains in minutes |
| SDF | "Signed distance field" | Function returning signed distance to the nearest surface; another implicit representation |

## 进一步阅读

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)变量变量分类器
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934)使3D复制照片成为神经网络问题
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989)哈希网,加快1000倍
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079)在生产中取代NeRF的架构
