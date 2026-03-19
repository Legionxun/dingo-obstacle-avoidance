# core/depth_estimation.py
import torch, cv2, os, sys
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision import transforms
from PIL import Image

# 添加MiDaS到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'intel-isl_MiDaS_master'))
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# 初始化训练中断标志
training_interrupted = False

# 添加全局变量用于训练进度更新
training_progress_callback = None

def set_training_progress_callback(callback):
    """设置训练进度回调函数"""
    global training_progress_callback
    training_progress_callback = callback


class MidasDepthEstimation(nn.Module):
    """深度估计模型"""
    def __init__(self, model_path=f"{project_root}/pre-training/dpt_large_384.pt"):
        super(MidasDepthEstimation, self).__init__()
        
        # 初始化MiDaS模型
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        
        # 设置为评估模式
        self.model.eval()
        
        # 定义预处理变换
        self.transform = transforms.Compose([
            Resize(
                width=384,
                height=384,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ])
        
    def forward(self, x):
        """进行深度估计"""
        if isinstance(x, Image.Image):
            x = np.array(x)
            if len(x.shape) == 3:
                x = x[:, :, ::-1]  # BGR到RGB
                
        if isinstance(x, np.ndarray):
            # 应用预处理变换
            sample = self.transform({"image": x})["image"]
            # 转换为张量
            input_batch = torch.from_numpy(sample).unsqueeze(0)
        else:
            # 假设已经是适当的张量格式
            input_batch = x
            
        # 进行深度估计
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # 调整输出大小以匹配输入
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=x.shape[:2] if isinstance(x, np.ndarray) else x.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        return prediction


class DepthFeatureExtractor(nn.Module):
    """深度特征提取模型"""
    def __init__(self, backbone='resnet50'):
        super(DepthFeatureExtractor, self).__init__()
        
        # 使用ResNet作为骨干网络
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            # 加载预训练权重
            if os.path.exists(f"{project_root}/pre-training/resnet50.pth"):
                self.backbone.load_state_dict(torch.load(f"{project_root}/pre-training/resnet50.pth"))
            self.feature_dim = 2048
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=False)
            # 加载预训练权重
            if os.path.exists(f"{project_root}/pre-training/densenet121.pth"):
                self.backbone.load_state_dict(torch.load(f"{project_root}/pre-training/densenet121.pth"))
            self.feature_dim = 1024
        else:
            raise ValueError("Unsupported backbone")
            
        # 移除最后的全连接层
        if backbone == 'resnet50':
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == 'densenet121':
            self.backbone = self.backbone.features
            # 添加自适应平均池化
            self.backbone = nn.Sequential(
                self.backbone,
                nn.AdaptiveAvgPool2d((1, 1))
            )
        
        # 深度特征处理层
        self.depth_processor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # 深度特征编码器
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # 特征适配器 - 确保特征尺寸一致
        self.rgb_adapter = nn.AdaptiveAvgPool2d((1, 1))
        self.depth_adapter = nn.AdaptiveAvgPool2d((1, 1))
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 256, 1024),  # 适配后的特征维度
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
    def forward(self, rgb_image, depth_map):
        """提取并融合RGB和深度特征"""
        # 确保输入张量具有相同的批次大小
        batch_size = rgb_image.size(0)
        
        # RGB图像特征提取
        rgb_features = self.backbone(rgb_image)
        # 适配RGB特征
        rgb_features = self.rgb_adapter(rgb_features)
        rgb_features = rgb_features.view(batch_size, -1)
        
        # 深度图特征提取 - 确保深度图是正确的格式
        if depth_map.size(1) == 1:  # 如果是单通道深度图
            # 将单通道深度图复制为三通道
            depth_map = depth_map.repeat(1, 3, 1, 1)
        elif depth_map.size(1) != 3:  # 如果不是三通道，调整为三通道
            depth_map = depth_map[:, :3, :, :] if depth_map.size(1) > 3 else torch.cat([depth_map, torch.zeros_like(depth_map[:, :3-depth_map.size(1), :, :])], dim=1)
        
        # 确保深度图和RGB图像具有相同的批次大小
        if depth_map.size(0) != batch_size:
            if depth_map.size(0) == 1 and batch_size > 1:
                # 如果深度图批次大小为1，而RGB图像批次大小大于1，则复制深度图
                depth_map = depth_map.repeat(batch_size, 1, 1, 1)
            elif depth_map.size(0) > batch_size:
                # 如果深度图批次大小大于RGB图像，则截取
                depth_map = depth_map[:batch_size]
        
        depth_features = self.depth_processor(depth_map)
        depth_features = self.depth_encoder(depth_features)
        # 适配深度特征
        depth_features = self.depth_adapter(depth_features)
        depth_features = depth_features.view(batch_size, -1)
        
        # 特征融合
        combined_features = torch.cat((rgb_features, depth_features), dim=1)
        fused_features = self.fusion(combined_features)
        
        return fused_features

class RiskAssessmentNetwork(nn.Module):
    """深度风险评估网络"""
    def __init__(self, input_dim=512, hidden_dim=256):
        super(RiskAssessmentNetwork, self).__init__()
        
        # 风险评估MLP
        self.risk_assessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3),  # 3个风险等级: 低、中、高
        )
        
    def forward(self, x):
        """评估风险"""
        risk_scores = self.risk_assessor(x)
        return risk_scores


class DepthRiskAssessmentSystem(nn.Module):
    """集成深度估计和风险评估的系统"""
    def __init__(self):
        super(DepthRiskAssessmentSystem, self).__init__()
        
        # MiDaS深度估计模型
        try:
            self.depth_estimator = MidasDepthEstimation(f"{project_root}/pre-training/dpt_large_384.pt")
        except Exception as e:
            print(f"无法初始化MiDaS模型: {e}")
            # 创建一个简化版本
            self.depth_estimator = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        
        # 特征提取器
        self.feature_extractor = DepthFeatureExtractor('resnet50')
        
        # 风险评估网络
        self.risk_assessor = RiskAssessmentNetwork()
        
    def forward(self, rgb_image):
        """前向传播: 输入RGB图像，输出风险评分和深度图"""
        # 估计深度图
        try:
            depth_map = self.depth_estimator(rgb_image)
        except Exception as e:
            print(f"深度估计出错: {e}")
            # 使用简化方法生成深度图
            depth_map = torch.mean(rgb_image, dim=1, keepdim=True)
        
        # 确保深度图为正确的维度
        if len(depth_map.shape) == 2:
            depth_map = depth_map.unsqueeze(0)
        if len(depth_map.shape) == 3:
            depth_map = depth_map.unsqueeze(0)
            
        # 特征提取和风险评估
        features = self.feature_extractor(rgb_image, depth_map)
        risk_scores = self.risk_assessor(features)
        
        return risk_scores, depth_map

def preprocess_image(image_path):
    """加载并预处理图像"""
    # 图像预处理变换
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载图像
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
        
    # 应用变换
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
    
    return input_batch, image


class KITTIDepthDataset(torch.utils.data.Dataset):
    """KITTI数据集用于深度风险评估"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 遍历数据
        image_dir = os.path.join(root_dir, 'training', 'image_2')
        label_dir = os.path.join(root_dir, 'training', 'label_2')
        
        if os.path.exists(image_dir):
            # 获取所有图像文件
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, img_file.replace('.png', '.txt'))
                self.samples.append((img_path, label_path))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        img_path, label_path = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 解析标签
        risk_level = self._parse_label(label_path)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, risk_level
    
    def _parse_label(self, label_path):
        # 解析KITTI标签文件，生成风险等级
        if not os.path.exists(label_path):
            return 0  # 默认低风险
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # 统计各类别数量
        car_count = 0
        pedestrian_count = 0
        cyclist_count = 0
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
                
            obj_type = parts[0]
            if obj_type == 'Car' or obj_type == 'Van' or obj_type == 'Truck':
                car_count += 1
            elif obj_type == 'Pedestrian' or obj_type == 'Person_sitting':
                pedestrian_count += 1
            elif obj_type == 'Cyclist':
                cyclist_count += 1
                
        # 根据检测到的对象数量返回风险等级
        # 0: 低风险, 1: 中风险, 2: 高风险
        total_objects = car_count + pedestrian_count + cyclist_count
        if total_objects == 0:
            return 0  # 低风险
        elif total_objects <= 2:
            return 1  # 中风险
        else:
            return 2  # 高风险

def train_depth_risk_model(num_epochs=15, progress_callback=None):
    """训练深度风险评估模型"""
    # 添加全局变量来支持训练中断
    global training_interrupted
    training_interrupted = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset_path = os.path.join(project_root, 'dataset', 'kitti_object')
    if not os.path.exists(dataset_path):
        print("请先解压KITTI数据集到指定目录")
        return
    
    train_dataset = KITTIDepthDataset(root_dir=dataset_path, transform=transform)
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        print("数据集为空，使用模拟数据集进行训练...")
        # 创建模拟数据集
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, size=500):
                self.size = size
                
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # 生成模拟数据
                image = torch.randn(3, 384, 384)
                # 风险等级: 0-低风险, 1-中风险, 2-高风险
                risk_level = torch.randint(0, 3, (1,)).item()
                return image, risk_level
        
        train_dataset = MockDataset(500)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 创建模型
    model = DepthRiskAssessmentSystem()
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 计算总步数用于进度计算
    total_steps = num_epochs * len(train_loader)
    
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            try:
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
            except Exception as e:
                print(f"训练过程中出错: {e}")
                continue
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算当前步骤和进度
            current_step = epoch * len(train_loader) + i + 1
            progress = int((current_step / total_steps) * 100)
            
            # 调用进度回调函数
            if progress_callback:
                progress_callback(progress, f"深度风险评估训练中 - Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}")

            # 打印进度
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 30:.4f}, Accuracy: {accuracy:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

            if training_interrupted:
                print("训练已被用户中断")
                return
        
        # 调整学习率
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Learning rate: {scheduler.get_last_lr()[0]:.6f}')
    
    print('深度风险评估模型训练完成')
    
    # 保存模型到models目录
    model_path = os.path.join(project_root, 'models', 'depth_risk_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存到: {model_path}')


"""if __name__ == "__main__":
    # 训练深度风险评估模型
    train_depth_risk_model()"""
