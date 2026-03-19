# core/motion_prediction.py
import torch, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 初始化训练中断标志
training_interrupted = False

# 添加全局变量用于训练进度更新
training_progress_callback = None

def set_training_progress_callback(callback):
    """设置训练进度回调函数"""
    global training_progress_callback
    training_progress_callback = callback


class MotionPredictionLSTMTransformer(nn.Module):
    """LSTM + Transformer 用于运动预测的模型"""
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, num_heads=8, num_classes=3, seq_length=10):
        super(MotionPredictionLSTMTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # LSTM编码器用于序列建模
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Transformer编码器用于全局依赖建模
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 位置编码
        self.pos_encoding = self._create_position_encoding(hidden_dim, seq_length)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def _create_position_encoding(self, d_model, max_len):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        """前向传播"""
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        
        # LSTM编码
        lstm_out, _ = self.lstm(x)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :lstm_out.size(1), :].to(x.device)
        transformer_input = lstm_out + pos_enc
        
        # Transformer编码
        transformer_out = self.transformer_encoder(transformer_input)
        
        # 使用最后一个时间步的输出进行分类
        output = self.fc(transformer_out[:, -1, :])
        return output


class KITTITrackingDataset(Dataset):
    """KITTI跟踪数据集用于运动预测"""
    def __init__(self, root_dir, seq_length=10, transform=None):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.transform = transform
        self.sequences = []
        
        # 遍历所有序列
        seq_dirs = os.listdir(os.path.join(root_dir, "training", 'image_02'))
        seq_dirs = [d for d in seq_dirs if os.path.isdir(os.path.join(root_dir, "training", 'image_02', d))]
        seq_dirs.sort()
        
        # 读取标注文件
        for seq_dir in seq_dirs:
            label_path = os.path.join(root_dir, "training", 'label_02', seq_dir + '.txt')
            if os.path.exists(label_path):
                self._parse_sequence(seq_dir, label_path)
                
    def _parse_sequence(self, seq_dir, label_path):
        """解析单个序列的标注文件，提取轨迹数据"""
        # 解析序列标注文件
        tracks = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
                
            frame_id = int(parts[0])
            track_id = int(parts[1])
            obj_class = parts[2]
            truncation = float(parts[3])
            occlusion = int(parts[4])
            obs_angle = float(parts[5])
            x1, y1, x2, y2 = map(float, parts[6:10])
            h, w, l = map(float, parts[10:13])
            tx, ty, tz = map(float, parts[13:16])
            ry = float(parts[16])
            
            # 只考虑主要对象类别
            if obj_class not in ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist']:
                continue
                
            # 计算中心点和大小
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            if track_id not in tracks:
                tracks[track_id] = []
                
            tracks[track_id].append({
                'frame_id': frame_id,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'class': obj_class
            })
            
        # 构造序列数据
        for track_id, track_data in tracks.items():
            track_data.sort(key=lambda x: x['frame_id'])
            
            # 提取特征序列
            if len(track_data) >= self.seq_length:
                for i in range(len(track_data) - self.seq_length + 1):
                    sequence = track_data[i:i+self.seq_length]
                    features = []
                    for item in sequence:
                        # 特征: [center_x, center_y, width, height]
                        features.append([item['center_x'], item['center_y'], item['width'], item['height']])
                    
                    # 确定对象类别标签
                    obj_class = sequence[-1]['class']
                    if obj_class in ['Car', 'Van', 'Truck']:
                        label = 2  # 车辆
                    elif obj_class in ['Pedestrian', 'Person_sitting']:
                        label = 1  # 行人
                    elif obj_class == 'Cyclist':
                        label = 0  # 骑行者
                    else:
                        label = 0
                        
                    self.sequences.append({
                        'features': features,
                        'label': label
                    })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        features = torch.tensor(sequence['features'], dtype=torch.float32)
        label = torch.tensor(sequence['label'], dtype=torch.long)
        return features, label

def train_motion_prediction(num_epochs=100, progress_callback=None):
    """训练运动预测模型"""
    # 添加全局变量来支持训练中断
    global training_interrupted
    training_interrupted = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据集
    dataset_path = os.path.join(project_root, 'dataset', 'kitti_tracking')
    if not os.path.exists(dataset_path):
        print("请先解压KITTI跟踪数据集到指定目录")
        return
    
    train_dataset = KITTITrackingDataset(root_dir=dataset_path, seq_length=10)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    # 创建模型
    model = MotionPredictionLSTMTransformer(input_dim=4, hidden_dim=128, num_layers=2, num_heads=8, num_classes=3, seq_length=10)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 计算总步数用于进度计算
    total_steps = num_epochs * len(train_loader)
    
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
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
                progress_callback(progress, f"运动预测训练中 - Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}")

            # 打印进度
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 50:.4f}, Accuracy: {accuracy:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0
                
            # 检查是否需要中断训练
            if training_interrupted:
                print("训练已被用户中断")
                return
        
        # 调整学习率
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Learning rate: {scheduler.get_last_lr()[0]:.6f}')
    
    print('运动预测模型训练完成')
    
    # 保存模型到models目录
    model_path = os.path.join(project_root, 'models', 'motion_prediction_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存到: {model_path}')


"""if __name__ == '__main__':
    # 调用函数进行训练
    train_motion_prediction()"""
