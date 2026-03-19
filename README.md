# 叮丁多模态深度视觉避障系统 (DingDing Multi-modal Depth Vision Obstacle Avoidance System)

<div align="center">
  <img src="img/icon.ico" alt="logo" width="128"/>
  <h3>基于深度学习的多模态机器人避障系统</h3>
  <p>集成障碍物检测、运动预测、深度估计与避让决策的智能视觉系统</p>
</div>

---

## 📖 简介

叮丁多模态深度视觉避障系统是一款基于计算机视觉和深度学习技术的智能避障解决方案。系统通过深度相机捕获环境信息，结合先进的机器学习算法，实现对障碍物的实时检测、运动轨迹预测、风险评估和避让决策。可广泛应用于服务机器人、自动驾驶车辆、智能监控等领域。

**核心架构**：采用 CNN-Swin Transformer、LSTM-Transformer、MiDaS 等前沿模型，通过多模态特征融合实现高精度、低延迟的避障决策。

---

## ✨ 主要特性

- **障碍物检测**：基于 CNN-Swin Transformer 混合模型，精准识别行人、车辆等障碍物（3 分类）。
- **运动预测**：LSTM-Transformer 混合模型，预测动态物体的未来轨迹（骑行者/行人/车辆分类）。
- **深度估计与风险评估**：集成 MiDaS 深度模型，结合 RGB 图像进行像素级深度估计，并输出低/中/高三级风险。
- **避让决策**：多特征融合神经网络，综合障碍物、运动、风险信息，输出紧急制动、转向、加速、保持等 5 类动作。
- **图形化界面**：基于 Tkinter 的现代化 GUI，支持图像/视频处理、实时摄像头、模型训练可视化。
- **实时处理**：支持 USB 摄像头实时推理，帧率可达 10+ FPS（取决于硬件）。
- **模块化设计**：各子模型可独立训练和替换，便于二次开发。

---

## 💻 系统要求

- **操作系统**：Windows 10/11 / Ubuntu 20.04+
- **Python**：3.8 ~ 3.10
- **硬件**：
  - 推荐：NVIDIA GPU (CUDA 11.3+) 用于训练和加速推理
  - 最低：CPU (Intel i5 或同等) 可运行基本功能
- **内存**：≥ 8 GB
- **存储**：≥ 10 GB（包含预训练模型和数据集）

---

## 🔧 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/yourname/DingDing_Obstacle_Avoidance.git
cd DingDing_Obstacle_Avoidance
```

### 2. 创建虚拟环境（可选）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载预训练模型
将以下预训练权重放入 `pre-training/` 目录：

- **MiDaS (DPT-Large)**：[dpt_large_384.pt](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt)  
- **ResNet50**：[resnet50.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)  
- **DenseNet121**：[densenet121.pth](https://download.pytorch.org/models/densenet121-63900597.pth) （可选）

### 5. 准备数据集（可选，用于训练）
系统支持 [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/) 进行模型训练。请下载以下子集并解压至 `dataset/` 目录：

- **KITTI Object Detection**：用于障碍物检测训练  
- **KITTI Tracking**：用于运动预测训练  
- **KITTI Raw / Depth**：用于深度风险评估训练（也可使用预生成标签）

目录结构示例：
```
dataset/
├── kitti_object/               # 障碍物检测
│   ├── training/
│   │   ├── image_2/
│   │   └── label_2/
│   └── testing/
├── kitti_tracking/              # 运动预测
│   ├── training/
│   │   ├── image_02/
│   │   └── label_02/
│   └── testing/
└── kitti_depth/                 # 深度估计（可选）
```

---

## 🚀 快速开始

### 启动 GUI
```bash
python DingDing_Multi-modal_Depth_Vision_Obstacle_Avoidance_System.py
```

### 使用流程

1. **加载测试图像**：点击左侧“加载测试图像”按钮，选择一张图片。
2. **处理图像**：点击“处理图像”，系统将依次运行障碍物检测、深度估计、风险判断和避让决策，结果会显示在右侧标签页。
3. **实时摄像头**：点击“开启摄像头”，可实时查看检测结果；点击“实时深度估计”可单独运行深度风险可视化。
4. **模型训练**：若缺失模型文件，系统启动时会提示训练。也可通过“训练”菜单或左侧按钮手动启动各模块训练。
5. **查看日志**：下方日志区域实时输出系统运行信息。

---

## 🧠 模型训练

### 单模块训练
- 障碍物检测：`train_obstacle_detection()`
- 运动预测：`train_motion_prediction()`
- 深度风险评估：`train_depth_risk_model()`
- 避让决策：`train_evasion_model()`

可通过 GUI 点击对应按钮，或在命令行中调用：
```python
from core.obstacle_detection import train_obstacle_detection
train_obstacle_detection(progress_callback=print)
```

### 全部重新训练
GUI 中点击“全部重新训练”或调用：
```python
from core.core_processing import IntegratedRobotSystem
system = IntegratedRobotSystem()
system.train_missing_models()  # 自动训练缺失模型
```

训练好的模型将保存在 `models/` 目录。

---

## 📁 文件结构

```
DingDing_Obstacle_Avoidance/
├── core/
│   ├── obstacle_detection.py      # 障碍物检测模型与训练
│   ├── motion_prediction.py        # 运动预测模型与训练
│   ├── depth_estimation.py         # 深度估计与风险评估
│   ├── evasion_decision.py         # 避让决策模型与训练
│   └── core_processing.py          # 系统集成与主流程
├── models/                         # 存放训练好的模型权重
├── pre-training/                   # 第三方预训练权重
├── dataset/                        # 数据集（用户自行放置）
├── logs/                           # 运行日志
├── img/                             # 图标等资源
├── DingDing_Multi-modal_Depth_Vision_Obstacle_Avoidance_System.py  # 主程序
├── requirements.txt                # Python依赖
└── README.md                       # 本文件
```

---

## 📦 依赖库

主要依赖如下（完整列表见 `requirements.txt`）：

- torch >= 1.10.0
- torchvision >= 0.11.0
- opencv-python >= 4.5.5
- numpy >= 1.21.0
- Pillow >= 9.0.0
- matplotlib >= 3.5.0
- pandas >= 1.4.0
- scikit-learn >= 1.0.0
- tkinter（Python 标准库）

---

## 🤝 贡献指南

欢迎通过 Issue 或 Pull Request 贡献代码。请确保：

- 代码风格符合 PEP 8。
- 新功能包含必要的注释和文档。
- 涉及模型修改时，提供训练结果对比。

---

## 📄 许可证

本项目采用 **MIT 许可证**。详情参见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 [MiDaS](https://github.com/isl-org/MiDaS) 提供的深度估计模型。
- 感谢 KITTI 数据集提供方。
- 感谢 PyTorch 和 OpenCV 社区。

---

**开发团队**：浙工大叮丁车队  
**开发者**：吴迅  
**版权所有 © 2025 浙工大叮丁车队**