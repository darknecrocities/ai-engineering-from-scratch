# 从零开始检测到物体

> 检测是分类加回,在特征地图中的每个位置运行,然后通过非最大压缩来清理.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 05 (Transfer Learning)
**Time:** ~75 minutes

## 学习目标

- 解释了和的设计,使检测成为密集的预测问题,并说明输出子中的每个数字意味着什么
- 计算盒子之间的交叉关系,从零开始实现非最大的抑制
- 在训练前的脊椎上建立一个最小的YOLO风格头,包括分类,对象性和盒子回归损失
- 阅读检测测量表行 (精度@0.5,回忆,mAP@0.5,mAP@0.5:0.95) 并选择下一个按

## 问题

分类说:"这个图像是狗".检测说"在像素 (112, 40, 280, 210),在图像 (400, 180, 560, 310),还有猫. "这个结构变化 预测每张图像的标签而不是一个标签的变量盒子数量 是每个自主系统,每个监控产品,每个文件布局解析器,以及每个工厂视觉线都取决于什么.

检测也是每一个工程的视觉交易都会出现一次. 你想要准确的框 (回归头),你想要对每个框 (分类头) 的正确类别,你想要模型知道什么时候没有什么可以检测 (对象性分数),你想要每一个真实对象的预测 (非最大压制). 错过任何一个,管道要么错过物体,报告幻觉的盒子,或者预测相同的物体15次,

2016年,YOLO (You Only Look Once, Redmon et al. 2016) 是通过单个向前传输一个网来实现所有这些实时运行的设计,同样的结构决定仍然是现代探测器 (YOLOv8,YOLOv9,YOLO-NAS,RT-DETR) 的脊柱.

## 概念

### 检测作为密集预测

一个分类器输出每张图像的C号码. 一个YOLO式检测器输出.`(S x S x (5 + C))`图像的数量,其中S是空间网格的尺寸.

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

它们中的每一个`S * S`电网电池预测`B`盒子,每盒子:

- 描述几何学的4个数字:`tx, ty, tw, th`现在,我们要去.
- 个数是对象性分数:"这个细胞中是否有一个中心的对象?"
- 数是类概率.

总数每一个细胞:`B * (5 + C)`对于VOC与`S=13, B=2, C=20`它们的数量是50个.

### 为什么网和

简单的回归预测`(x, y, w, h)`对于每个对象来说,这是一个绝对坐标.对于一个 conv 网络来说,这是很难的,因为翻译图像不应该翻译所有预测的相同数量.每个对象是空间.

结解决了第二个问题.一个3×3 conv不能轻松退回500像素宽的盒子从16像素接收场特征细胞.`B`模型学会选择正确的,并推进它,而不是从无处退回.

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

现代探测器通常使用FPN,每个分辨率的具不同具. 低分辨率地图上的小具,低分辨率地图上的大具.

### 解码预测

的`tx, ty, tw, th`它们不是框坐标,而是在绘图之前要转换的回归目标:

```
centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)
```

`sigmoid`保持细胞内部的中心偏移. `exp`允许宽度从杆自由,而不需要标志转换.`stride`解码步骤是从2版本以来的每一个YOLO版本相同的.

### 其他

检测两个盒子之间的普遍相似度指标:

```
IoU(A, B) = area(A intersect B) / area(A union B)
```

预测和基础真相框之间的 IoU 是决定一个预测是否算为真正 (通常是 IoU >=0.5). 两个预测之间的 IoU 是NMS用来分倍的.

### 超出最大压力

基于相邻的杆训练的 conv网络通常会预测同一对象的重叠框.NMS保持最高可靠性预测,并删除任何其他预测,如果 IoU 超过门值.

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

对于对象检测,典型门值为0.45. 最近的检测器取代了标准NMS的`soft-NMS`现在`DIoU-NMS`它们的结构性目的是相同的.

### 损失

罗损失是三损失加重:

```
L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)
```

只有包含物体的细胞导致了盒子回归和分类损失. 没有物体的细胞只导致了物体性损失 (教模型保持沉默). `lambda_noobj`由于绝大多数细胞是空的,否则将占据总损失的主导地位.

现代变体将MSE盒损失换为CIoU/DIoU (直接优化IoU),使用焦失为类失衡,并平衡对象性与质量焦失.

### 检测指标

准确性不会转移到检测.

- **Precision@IoU=0.5**的预测被认为是正确的,
- **Recall@IoU=0.5**,我们发现了多少的真实物体.
- **AP@0.5**精度回忆曲线面积在IOU门值为0.5;每类一个数字.
- **mAP@0.5:0.95**平均AP超过IOU门值0.5,0.55, ...,0.95.

报告四个.在mAP@0.5上强,但在mAP@0.5:0.95上弱的探测器,大致地定位,但不紧密;更好的盒子回归损失;高精度和低回忆的探测器过于保守;降低信任门或增加对象重量.

```figure
object-detection-nms
```

## 建立它

### 步骤1:

整个课程的工作马. 在两个盒子上工作.`(x1, y1, x2, y2)`格式

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回一个`(N_a, N_b)`通过使一个数组形状,使用它对抗一个单一的地面真相框.`(1, 4)`现在,我们要去.

### 步骤2:非最大压缩

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

确定性主义者`O(N log N)`,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,`torchvision.ops.nms`在相同的输入.

### 步骤3: 框编码和解码

转换像素坐标和`(tx, ty, tw, th)`网络实际上会退缩.

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试:编码一个框,然后解码 你应该得到回来非常接近原始的东西 (直到sigmoid反向不完全可逆时`tx`没有在sigmoid后的范围中).

### 步骤4:最小的YOLO头

图片的1x1集,重塑为`(B, S, S, num_anchors, 5 + C)`现在,我们要去.

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

输出形状:`(N, H, W, num_anchors, 5 + C)`最后一个维度是坚持的`[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`现在,我们要去.

### 第五步: 实践真相

对于每一个基本真理盒子,`(cell, anchor)`负责.

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

轮选择是"最好的形状 IoU 与地面真相"一个价格便宜的代理,与YOLOv2/v3任务匹配. v5及后者使用更复杂的策略 (任务一致匹配,动态 k) 来完善相同的想法.

### 步骤 6: 三次损失

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

五个超级参数,每个YOLO教程都会硬码或扫描.`lambda_coord=5, lambda_noobj=0.5`像原始YOLOv1纸样,仍然是合理的默认.

### 步骤7: 推进管道

解码原始头输出,应用sigmoid/exp,对象性门和NMS.

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这就是完整的评估路径:头 -> 解码 -> 门 -> NMS.

## 用它

`torchvision.models.detection`对于预训练模型,需要三行运载.

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对于实时推断管道,`ultralytics`标准是:`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`模型内部处理解码和NMS,并返回相同的信息.`boxes / scores / labels`你在上面建造的三倍.

## 运送它

这一课产生了:

- `outputs/prompt-detection-metric-reader.md`一个提示,转换一个`precision, recall, AP, mAP@0.5:0.95`排列一行诊断,然后再进行一个最有用的实验.
- `outputs/skill-anchor-designer.md`一个技能,在基础真相框的数据集中,运行k-means`(w, h)`返回每个FPN级别的杆设置,加上需要选择正确的杆数量的覆盖统计数据.

## 运动

1. **(Easy)**实施`box_iou`击它.`torchvision.ops.box_iou`检查最大绝对差距在以下`1e-6`现在,我们要去.
2. **(Medium)**港口`yolo_loss`通过使用`CIoU`在100图像合成数据集上显示,CIoU在同一时间段相比,与MSE相比,接近更好的最终mAP@0.5:0.95.
3. **(Hard)**实现多尺度推理:通过模型以三分辨率输送相同的图像,结合盒子预测,并在结尾运行单个NMS. 在一个持久的集合上测量mAP升级与单尺度推理.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Anchor | "Box prior" | A pre-defined box shape at each grid cell from which the network predicts deltas instead of absolute coordinates |
| IoU | "Overlap" | Intersection-over-union of two boxes; the universal similarity measure in detection |
| NMS | "Deduplicate" | Greedy algorithm that keeps highest-score predictions and removes overlapping ones above a threshold |
| Objectness | "Is there something here" | Per-anchor, per-cell scalar predicting whether an object is centred in that cell |
| Grid stride | "Downsample factor" | Pixels per grid cell; a 416-px input with a 13-grid head has stride 32 |
| mAP | "Mean average precision" | Average of the area under the precision-recall curve, averaged over classes and (for COCO) IoU thresholds |
| AP@0.5 | "PASCAL VOC AP" | Average precision with IoU threshold 0.5; the lenient version of the metric |
| mAP@0.5:0.95 | "COCO AP" | Average over IoU thresholds 0.5..0.95 step 0.05; the strict version and current community standard |

## 进一步阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640)创建纸;从那以后的每一个YOLO都是这种结构的完善
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767)引入了多尺度FPN式头的纸;仍然是最清晰的图表
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com)目前的生产参考;涵盖数据集格式,增强,培训食谱
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904)完整的检测动物园的最佳普通英语游览; 对于了解DETR,RetinaNet,FCOS和YOLO的关系,是无价的
