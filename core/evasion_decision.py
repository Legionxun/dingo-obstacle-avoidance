# core/evasion_decision.py
import torch, os, cv2, sys
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from obstacle_detection import ObstacleDetectionCNN_SwinTransformer
from motion_prediction import MotionPredictionLSTMTransformer
from depth_estimation import DepthRiskAssessmentSystem

# 项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加中文字体支持
plt.rc("font", family='Microsoft YaHei')
plt.rcParams['axes.unicode_minus'] = False

# 训练中断标志
training_interrupted = False


class EvasionDecisionDataset(Dataset):
    """避让决策数据集"""
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform

        # 读取标签文件
        self.labels_df = pd.read_csv(label_file, sep=' ', header=None,
                                     names=['frame', 'steering_angle', 'acceleration', 'brake', 'gear'])

        # 获取所有图像文件
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 确保图像文件数量与标签数量匹配
        if len(self.image_files) != len(self.labels_df):
            print(f"警告: 图像数量({len(self.image_files)})与标签数量({len(self.labels_df)})不匹配")

        # 取较小值
        self.data_size = min(len(self.image_files), len(self.labels_df))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取对应的标签数据
        label_data = self.labels_df.iloc[idx]
        steering_angle = label_data['steering_angle']
        acceleration = label_data['acceleration']
        brake = label_data['brake']
        gear = label_data['gear']

        # 根据标签数据推导避让动作
        action_label = self._derive_action(steering_angle, acceleration, brake, gear)

        return image, action_label

    def _derive_action(self, steering_angle, acceleration, brake, gear):
        """
        根据驾驶数据推导避让动作
        0: 紧急制动 - 前方有近距障碍物
        1: 向右转向 - 右侧有可通行空间
        2: 向左转向 - 左侧有可通行空间
        3: 加速 - 道路畅通
        4: 保持 - 正常行驶
        """
        # 紧急制动判断
        if brake > 0.5 or (acceleration < -0.3 and abs(steering_angle) < 0.1):
            return 0  # 紧急制动

        # 转向判断
        if steering_angle > 0.2:  # 转向角大于阈值，向右转
            return 1  # 向右转向
        elif steering_angle < -0.2:  # 转向角小于阈值，向左转
            return 2  # 向左转向

        # 加速判断
        if acceleration > 0.3 and abs(steering_angle) < 0.1:
            return 3  # 加速

        # 保持正常行驶
        return 4  # 保持


class FeatureExtractor:
    """特征提取器"""
    def __init__(self, device):
        self.device = device

        # 初始化各子模型
        self.obstacle_detector = ObstacleDetectionCNN_SwinTransformer(num_classes=3).to(device)
        self.motion_predictor = MotionPredictionLSTMTransformer(
            input_dim=4, hidden_dim=128, num_layers=2, num_heads=8, num_classes=3, seq_length=10).to(device)
        self.depth_estimator = DepthRiskAssessmentSystem().to(device)

        # 加载权重
        self._load_models()

        # 设置为评估模式
        self.obstacle_detector.eval()
        self.motion_predictor.eval()
        self.depth_estimator.eval()

        # 预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_models(self):
        """加载预训练模型权重"""
        obstacle_model_path = os.path.join(project_root, 'models', 'obstacle_detection_model.pth')
        motion_model_path = os.path.join(project_root, 'models', 'motion_prediction_model.pth')
        depth_model_path = os.path.join(project_root, 'models', 'depth_risk_model.pth')

        if os.path.exists(obstacle_model_path):
            self.obstacle_detector.load_state_dict(torch.load(obstacle_model_path, map_location=self.device))
        else:
            print(f"警告: 找不到障碍物检测模型 {obstacle_model_path}")

        if os.path.exists(motion_model_path):
            self.motion_predictor.load_state_dict(torch.load(motion_model_path, map_location=self.device))
        else:
            print(f"警告: 找不到运动预测模型 {motion_model_path}")

        if os.path.exists(depth_model_path):
            self.depth_estimator.load_state_dict(torch.load(depth_model_path, map_location=self.device))
        else:
            print(f"警告: 找不到深度风险模型 {depth_model_path}")

    def extract_features(self, image_tensor):
        """提取图像的综合特征"""
        with torch.no_grad():
            # 1. 障碍物检测特征
            obstacle_outputs = self.obstacle_detector(image_tensor.unsqueeze(0))
            obstacle_probs = torch.softmax(obstacle_outputs, dim=1)[0]

            # 2. 深度风险评估特征
            risk_outputs, _ = self.depth_estimator(image_tensor.unsqueeze(0))
            risk_probs = torch.softmax(risk_outputs, dim=1)[0]

            # 3. 运动预测特征
            motion_features = torch.randn(1, 10, 4).to(self.device)  # 使用随机特征模拟
            motion_outputs = self.motion_predictor(motion_features)
            motion_probs = torch.softmax(motion_outputs, dim=1)[0]

        return obstacle_probs, motion_probs, risk_probs


class EvasionDecisionNetwork(nn.Module):
    """综合避让决策网络"""
    def __init__(self, obstacle_dim=3, motion_dim=3, risk_dim=3, hidden_dim=256, num_actions=5):
        super(EvasionDecisionNetwork, self).__init__()
        self.obstacle_processor = nn.Sequential(
            nn.Linear(obstacle_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.motion_processor = nn.Sequential(
            nn.Linear(motion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.risk_processor = nn.Sequential(
            nn.Linear(risk_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.decision_layer = nn.Linear(hidden_dim // 2, num_actions)

    def forward(self, obstacle_features, motion_features, risk_features):
        obstacle_processed = self.obstacle_processor(obstacle_features)
        motion_processed = self.motion_processor(motion_features)
        risk_processed = self.risk_processor(risk_features)
        combined = torch.cat((obstacle_processed, motion_processed, risk_processed), dim=1)
        fused = self.fusion(combined)
        decision = self.decision_layer(fused)
        return decision


class EvasionDecisionTrainer:
    """避让决策模型训练器"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor(self.device)
        self.model = EvasionDecisionNetwork(num_actions=5).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_dataset(self):
        """创建训练数据集"""
        dataset_path = os.path.join(project_root, 'dataset', 'evasion', 'training')
        image_dir = os.path.join(dataset_path, 'image_2')
        label_file = os.path.join(dataset_path, 'label_2', 'data.txt')

        if not os.path.exists(image_dir) or not os.path.exists(label_file):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        dataset = EvasionDecisionDataset(image_dir, label_file, transform=self.transform)
        return dataset

    def train(self, num_epochs=50, batch_size=16, progress_callback=None):
        """训练避让决策模型"""
        global training_interrupted
        training_interrupted = False  # 重置中断标志

        print("创建训练数据集...")
        dataset = self.create_dataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f"数据集大小: {len(dataset)}")
        print(f"开始训练避让决策模型，共 {num_epochs} 轮...")

        best_accuracy = 0.0
        train_losses = []
        train_accuracies = []
        
        # 计算总步数用于进度计算
        total_steps = num_epochs * len(dataloader)

        for epoch in range(num_epochs):
            # 检查中断标志
            if training_interrupted:
                print("训练已被用户中断")
                break

            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                # 检查中断标志
                if training_interrupted:
                    print("训练已被用户中断")
                    break

                images, labels = images.to(self.device), labels.to(self.device)

                # 提取特征
                obstacle_features = []
                motion_features = []
                risk_features = []

                for i in range(images.size(0)):
                    # 检查中断标志
                    if training_interrupted:
                        break

                    obs_prob, mot_prob, risk_prob = self.feature_extractor.extract_features(images[i])
                    obstacle_features.append(obs_prob)
                    motion_features.append(mot_prob)
                    risk_features.append(risk_prob)

                # 如果被中断，跳出循环
                if training_interrupted:
                    break

                obstacle_features = torch.stack(obstacle_features).to(self.device)
                motion_features = torch.stack(motion_features).to(self.device)
                risk_features = torch.stack(risk_features).to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(obstacle_features, motion_features, risk_features)
                loss = self.criterion(outputs, labels)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 计算当前步骤和进度
                current_step = epoch * len(dataloader) + batch_idx + 1
                progress = int((current_step / total_steps) * 100)
                
                # 调用进度回调函数
                if progress_callback:
                    progress_callback(progress, f"避让决策训练中 - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}")

                # 每10个batch打印一次
                if batch_idx % 10 == 9:
                    batch_acc = 100. * correct / total
                    print(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx + 1}, '
                          f'Loss: {running_loss / 10:.4f}, Acc: {batch_acc:.2f}%')
                    running_loss = 0.0

            # 如果被中断，跳出epoch循环
            if training_interrupted:
                break

            # 计算epoch准确率
            epoch_acc = 100. * correct / total
            train_losses.append(running_loss / len(dataloader))
            train_accuracies.append(epoch_acc)

            print(f'Epoch {epoch + 1}/{num_epochs} completed. '
                  f'Train Acc: {epoch_acc:.2f}%, LR: {self.scheduler.get_last_lr()[0]:.6f}')

            # 保存最佳模型
            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                model_path = os.path.join(project_root, 'models', 'evasion_decision_model.pth')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print(f'新最佳模型已保存，准确率: {best_accuracy:.2f}%')

            # 更新学习率
            self.scheduler.step()

        if training_interrupted:
            print("训练已被用户中断")
        else:
            print(f'训练完成！最佳准确率: {best_accuracy:.2f}%')
        return train_losses, train_accuracies


class RealTimeEvasionSystem():
    """实时避让决策系统"""
    def __init__(self):
        super(RealTimeEvasionSystem, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化各子模型
        self.obstacle_detector = ObstacleDetectionCNN_SwinTransformer(num_classes=3).to(self.device)
        self.motion_predictor = MotionPredictionLSTMTransformer(
            input_dim=4, hidden_dim=128, num_layers=2, num_heads=8, num_classes=3, seq_length=10).to(self.device)
        self.depth_estimator = DepthRiskAssessmentSystem().to(self.device)
        self.evasion_decider = EvasionDecisionNetwork(num_actions=5).to(self.device)

        # 加载权重
        self._load_models()

        # 设置为评估模式
        self.obstacle_detector.eval()
        self.motion_predictor.eval()
        self.depth_estimator.eval()
        self.evasion_decider.eval()

        self.decision_labels = ['紧急制动', '向右转向', '向左转向', '加速', '保持']

        # 使用与训练时相同的预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_models(self):
        """加载所有模型权重"""
        models_dir = os.path.join(project_root, 'models')

        # 加载障碍物检测模型
        obstacle_model_path = os.path.join(models_dir, 'obstacle_detection_model.pth')
        if os.path.exists(obstacle_model_path):
            self.obstacle_detector.load_state_dict(torch.load(obstacle_model_path, map_location=self.device))
        else:
            print(f"警告: 找不到障碍物检测模型 {obstacle_model_path}")

        # 加载运动预测模型
        motion_model_path = os.path.join(models_dir, 'motion_prediction_model.pth')
        if os.path.exists(motion_model_path):
            self.motion_predictor.load_state_dict(torch.load(motion_model_path, map_location=self.device))
        else:
            print(f"警告: 找不到运动预测模型 {motion_model_path}")

        # 加载深度风险模型
        depth_model_path = os.path.join(models_dir, 'depth_risk_model.pth')
        if os.path.exists(depth_model_path):
            self.depth_estimator.load_state_dict(torch.load(depth_model_path, map_location=self.device))
        else:
            print(f"警告: 找不到深度风险模型 {depth_model_path}")

        # 加载避让决策模型
        evasion_model_path = os.path.join(models_dir, 'evasion_decision_model.pth')
        if os.path.exists(evasion_model_path):
            self.evasion_decider.load_state_dict(torch.load(evasion_model_path, map_location=self.device))
        else:
            print(f"警告: 找不到避让决策模型 {evasion_model_path}")

    def run_inference_on_image(self, image_path):
        """
        在单张图片上运行推理流程
        :param image_path: 图像路径
        :return: 决策标签和置信度
        """
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).to(self.device)

            with torch.no_grad():
                # 1. 障碍物检测
                obstacle_outputs = self.obstacle_detector(image_tensor.unsqueeze(0))
                obstacle_probs = torch.softmax(obstacle_outputs, dim=1)[0]

                # 2. 深度风险评估
                risk_outputs, _ = self.depth_estimator(image_tensor.unsqueeze(0))
                risk_probs = torch.softmax(risk_outputs, dim=1)[0]

                # 3. 运动预测
                motion_features = torch.randn(1, 10, 4).to(self.device)
                motion_outputs = self.motion_predictor(motion_features)
                motion_probs = torch.softmax(motion_outputs, dim=1)[0]

                # 4. 避让决策
                decision_logits = self.evasion_decider(
                    obstacle_probs.unsqueeze(0),
                    motion_probs.unsqueeze(0),
                    risk_probs.unsqueeze(0)
                )
                decision_probs = torch.softmax(decision_logits, dim=1)
                best_decision_idx = torch.argmax(decision_probs, dim=1).item()
                confidence = torch.max(decision_probs).item()

            return self.decision_labels[best_decision_idx], confidence

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            return "错误", 0.0

    def run_inference_on_video(self, video_path):
        """
        在视频上运行推理流程
        :param video_path: 视频路径
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 != 0:  # 每5帧处理一次，提高处理速度
                continue

            try:
                # 转换为PIL图像并预处理
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_tensor = self.transform(image).to(self.device)

                with torch.no_grad():
                    # 1. 障碍物检测
                    obstacle_outputs = self.obstacle_detector(image_tensor.unsqueeze(0))
                    obstacle_probs = torch.softmax(obstacle_outputs, dim=1)[0]

                    # 2. 深度风险评估
                    risk_outputs, _ = self.depth_estimator(image_tensor.unsqueeze(0))
                    risk_probs = torch.softmax(risk_outputs, dim=1)[0]

                    # 3. 运动预测
                    motion_features = torch.randn(1, 10, 4).to(self.device)
                    motion_outputs = self.motion_predictor(motion_features)
                    motion_probs = torch.softmax(motion_outputs, dim=1)[0]

                    # 4. 避让决策
                    decision_logits = self.evasion_decider(
                        obstacle_probs.unsqueeze(0),
                        motion_probs.unsqueeze(0),
                        risk_probs.unsqueeze(0)
                    )
                    decision_probs = torch.softmax(decision_logits, dim=1)
                    best_decision_idx = torch.argmax(decision_probs, dim=1).item()
                    confidence = torch.max(decision_probs).item()

                action = self.decision_labels[best_decision_idx]
                print(f"Frame {frame_count}: Action: {action}, Confidence: {confidence:.2f}")

                # 显示结果
                cv2.putText(frame, f"{action} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Evasion Decision", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"处理视频帧时出错: {str(e)}")
                continue

        cap.release()
        cv2.destroyAllWindows()

    def run_inference_on_frame(self, frame):
        """
        在单帧图像上运行推理流程（用于实时摄像头处理）
        :param frame: OpenCV图像帧
        :return: 决策标签和置信度
        """
        try:
            # 转换为PIL图像并预处理
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.transform(image).to(self.device)

            with torch.no_grad():
                # 1. 障碍物检测
                obstacle_outputs = self.obstacle_detector(image_tensor.unsqueeze(0))
                obstacle_probs = torch.softmax(obstacle_outputs, dim=1)[0]

                # 2. 深度风险评估
                risk_outputs, _ = self.depth_estimator(image_tensor.unsqueeze(0))
                risk_probs = torch.softmax(risk_outputs, dim=1)[0]

                # 3. 运动预测
                motion_features = torch.randn(1, 10, 4).to(self.device)
                motion_outputs = self.motion_predictor(motion_features)
                motion_probs = torch.softmax(motion_outputs, dim=1)[0]

                # 4. 避让决策
                decision_logits = self.evasion_decider(
                    obstacle_probs.unsqueeze(0),
                    motion_probs.unsqueeze(0),
                    risk_probs.unsqueeze(0)
                )
                decision_probs = torch.softmax(decision_logits, dim=1)
                best_decision_idx = torch.argmax(decision_probs, dim=1).item()
                confidence = torch.max(decision_probs).item()

            return self.decision_labels[best_decision_idx], confidence

        except Exception as e:
            print(f"处理图像帧时出错: {str(e)}")
            return "错误", 0.0


def train_evasion_model(progress_callback=None):
    """训练避让决策模型的主函数"""
    global training_interrupted
    print("开始训练避让决策模型...")
    trainer = EvasionDecisionTrainer()

    try:
        train_losses, train_accuracies = trainer.train(num_epochs=50, batch_size=8, progress_callback=progress_callback)
        if not training_interrupted:
            print("避让决策模型训练完成！")

            # 绘制训练曲线
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies)
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')

            plt.tight_layout()
            plt.savefig(os.path.join(project_root, 'models', 'evasion_training_curve.png'))
            plt.show()
        else:
            print("避让决策模型训练已被中断")
    except Exception as e:
        if training_interrupted:
            print("避让决策模型训练已被中断")
        else:
            print(f"训练过程中出错: {str(e)}")


"""if __name__ == "__main__":
    # 检查模型文件是否存在
    if not os.path.exists(os.path.join(project_root, 'models', 'evasion_decision_model.pth')):
        train_evasion_model()

    # 初始化避让决策系统
    print("初始化避让决策系统...")
    system = RealTimeEvasionSystem()

    # 查找图像和视频进行测试
    test_dir = os.path.join(project_root, "test")
    if not os.path.exists(test_dir):
        print("请将测试图像或视频放入 ./test 目录")
    else:
        print(f"在目录 {test_dir} 中查找测试文件...")
        files_processed = 0
        for file in os.listdir(test_dir):
            path = os.path.join(test_dir, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"[Image] 处理 {file}")
                label, conf = system.run_inference_on_image(path)
                print(f"[Image] {file} -> Action: {label}, Confidence: {conf:.2f}")
                files_processed += 1
            elif file.lower().endswith(('.mp4', '.avi', '.mov')):
                print(f"[Video] 处理 {file}")
                system.run_inference_on_video(path)
                files_processed += 1

        if files_processed == 0:
            print("未找到任何图像或视频文件，请确保文件格式正确并放在 ./test 目录中")"""
