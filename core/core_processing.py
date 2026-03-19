# core/core_processing.py
import torch, cv2, os, sys
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 导入各个模块
from obstacle_detection import ObstacleDetectionCNN_SwinTransformer, train_obstacle_detection
from motion_prediction import MotionPredictionLSTMTransformer, train_motion_prediction
from depth_estimation import DepthRiskAssessmentSystem, train_depth_risk_model
from evasion_decision import RealTimeEvasionSystem, train_evasion_model


class IntegratedRobotSystem:
    """集成机器人系统"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 记录需要训练的模型
        self.models_to_train = []
        
        # 初始化各个子系统
        self._init_obstacle_detection()
        self._init_motion_prediction()
        self._init_depth_estimation()
        self._init_evasion_decision()
        
    def _init_obstacle_detection(self):
        """初始化障碍物检测系统"""
        print("初始化障碍物检测系统...")
        self.obstacle_detector = ObstacleDetectionCNN_SwinTransformer(num_classes=3)

        # 检查障碍物决策模型文件是否存在
        model_path = os.path.join(project_root, 'models', 'obstacle_detection_model.pth')
        if not os.path.exists(model_path):
            print("障碍物检测模型文件缺失")
            self.models_to_train.append("obstacle")
            return

        try:
            self.obstacle_detector.load_state_dict(torch.load(model_path, map_location=self.device))
            print("成功加载障碍物检测模型")
            self.obstacle_detector = self.obstacle_detector.to(self.device)
            self.obstacle_detector.eval()
        except Exception as e:
            print(f"加载障碍物检测模型失败: {e}")
            self.models_to_train.append("obstacle")

    def _init_motion_prediction(self):
        """初始化运动预测系统"""
        print("初始化运动预测系统...")
        self.motion_predictor = MotionPredictionLSTMTransformer(
            input_dim=4, hidden_dim=128, num_layers=2, num_heads=8, num_classes=3, seq_length=10)

        # 检查运动决策模型文件是否存在
        model_path = os.path.join(project_root, 'models', 'motion_prediction_model.pth')
        if not os.path.exists(model_path):
            print("运动预测模型文件缺失")
            self.models_to_train.append("motion")
            return

        # 尝试加载模型，如果失败则标记需要重新训练
        try:
            self.motion_predictor.load_state_dict(torch.load(model_path, map_location=self.device))
            print("成功加载运动预测模型")
            self.motion_predictor = self.motion_predictor.to(self.device)
            self.motion_predictor.eval()
        except Exception as e:
            print(f"加载运动预测模型失败: {e}")
            self.models_to_train.append("motion")
        
    def _init_depth_estimation(self):
        """初始化深度估计和风险评估系统"""
        print("初始化深度估计系统...")
        self.depth_estimator = DepthRiskAssessmentSystem()

        # 检查深度估计模型文件是否存在
        model_path = os.path.join(project_root, 'models', 'depth_risk_model.pth')
        if not os.path.exists(model_path):
            print("深度风险评估模型文件缺失")
            self.models_to_train.append("depth")
            return

        # 尝试加载模型，如果失败则标记需要重新训练
        try:
            self.depth_estimator.load_state_dict(torch.load(model_path, map_location=self.device))
            print("成功加载深度风险评估模型")
            self.depth_estimator = self.depth_estimator.to(self.device)
            self.depth_estimator.eval()
        except Exception as e:
            print(f"加载深度风险评估模型失败: {e}")
            self.models_to_train.append("depth")

    def _init_evasion_decision(self):
        """初始化避让决策系统"""
        print("初始化避让决策系统...")
        self.evasion_decision_system = RealTimeEvasionSystem()

        # 检查避让决策模型文件是否存在
        model_path = os.path.join(project_root, 'models', 'evasion_decision_model.pth')
        if not os.path.exists(model_path):
            print("避让决策模型文件缺失")
            self.models_to_train.append("evasion")
            return

        # 尝试加载模型，如果失败则标记需要重新训练
        try:
            self.depth_estimator.load_state_dict(torch.load(model_path, map_location=self.device))
            print("成功加载避让决策模型")
            self.depth_estimator = self.depth_estimator.to(self.device)
            self.depth_estimator.eval()
        except Exception as e:
            print(f"加载避让决策模型失败: {e}")
            self.models_to_train.append("evasion")

        
    def train_missing_models(self):
        """训练缺失的模型"""
        trained_models = []
        for model_type in self.models_to_train:
            try:
                if model_type == "obstacle":
                    print("开始训练障碍物检测模型...")
                    train_obstacle_detection()
                    # 重新加载模型
                    model_path = os.path.join(project_root, 'models', 'obstacle_detection_model.pth')
                    self.obstacle_detector = ObstacleDetectionCNN_SwinTransformer(num_classes=3)
                    self.obstacle_detector.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.obstacle_detector = self.obstacle_detector.to(self.device)
                    self.obstacle_detector.eval()
                    trained_models.append("obstacle")
                    print("障碍物检测模型训练完成")
                    
                elif model_type == "motion":
                    print("开始训练运动预测模型...")
                    train_motion_prediction()
                    # 重新加载模型
                    model_path = os.path.join(project_root, 'models', 'motion_prediction_model.pth')
                    self.motion_predictor = MotionPredictionLSTMTransformer(
                        input_dim=4, hidden_dim=128, num_layers=2, num_heads=8, num_classes=3, seq_length=10)
                    self.motion_predictor.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.motion_predictor = self.motion_predictor.to(self.device)
                    self.motion_predictor.eval()
                    trained_models.append("motion")
                    print("运动预测模型训练完成")
                    
                elif model_type == "depth":
                    print("开始训练深度风险评估模型...")
                    train_depth_risk_model()
                    # 重新加载模型
                    model_path = os.path.join(project_root, 'models', 'depth_risk_model.pth')
                    self.depth_estimator = DepthRiskAssessmentSystem()
                    self.depth_estimator.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.depth_estimator = self.depth_estimator.to(self.device)
                    self.depth_estimator.eval()
                    trained_models.append("depth")
                    print("深度风险评估模型训练完成")
                elif model_type == "evasion":
                    print("开始训练避让决策模型...")
                    train_evasion_model()
                    self.evasion_decision_system = RealTimeEvasionSystem()
                    trained_models.append("evasion")
                    print("避让决策模型训练完成")
            except Exception as e:
                print(f"训练{model_type}模型时出错: {e}")
                
        # 从待训练列表中移除已训练的模型
        for model in trained_models:
            if model in self.models_to_train:
                self.models_to_train.remove(model)
                
        return len(trained_models) > 0  # 返回是否训练了模型
        
    def preprocess_image(self, image):
        """预处理图像"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
        return input_batch.to(self.device), image
    
    def detect_obstacles(self, image):
        """检测障碍物"""
        if not hasattr(self, 'obstacle_detector') or "obstacle" in self.models_to_train:
            # 使用默认值
            probabilities = np.array([0.33, 0.33, 0.34])
            predicted_class = 0
            confidence = 0.33
        else:
            input_batch, _ = self.preprocess_image(image)
            
            with torch.no_grad():
                outputs = self.obstacle_detector(input_batch)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
        # 类别映射
        class_labels = ['无障碍物', '行人', '车辆']
        obstacle_type = class_labels[predicted_class]
        
        print(f"障碍物检测结果: {obstacle_type} (置信度: {confidence:.4f})")
        return predicted_class, confidence, probabilities.cpu().numpy()[0] if isinstance(probabilities, torch.Tensor) else probabilities
    
    def predict_motion(self, tracking_sequence):
        """
        预测物体运动
        tracking_sequence 应该是一个包含连续帧中物体位置的序列
        格式: [[center_x, center_y, width, height], ...]
        """
        if not hasattr(self, 'motion_predictor') or "motion" in self.models_to_train:
            # 使用默认值
            probabilities = np.array([0.33, 0.33, 0.34])
            predicted_class = 1
            confidence = 0.33
        else:
            with torch.no_grad():
                sequence_tensor = torch.tensor(tracking_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                outputs = self.motion_predictor(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
        # 类别映射
        class_labels = ['骑行者', '行人', '车辆']
        motion_type = class_labels[predicted_class]
        
        print(f"运动预测结果: {motion_type} (置信度: {confidence:.4f})")
        return predicted_class, confidence, probabilities.cpu().numpy()[0] if isinstance(probabilities, torch.Tensor) else probabilities
    
    def estimate_depth_and_risk(self, image):
        """估计深度和风险"""
        if not hasattr(self, 'depth_estimator') or "depth" in self.models_to_train:
            # 使用默认值
            probabilities = np.array([0.33, 0.33, 0.34])
            predicted_risk = 1
            confidence = 0.33
            depth_map = None
        else:
            input_batch, _ = self.preprocess_image(image)
            
            with torch.no_grad():
                try:
                    risk_scores, depth_map = self.depth_estimator(input_batch)
                    probabilities = torch.softmax(risk_scores, dim=1)
                    predicted_risk = torch.argmax(probabilities, dim=1).item()
                    confidence = torch.max(probabilities).item()
                except Exception as e:
                    print(f"深度估计出错: {e}")
                    probabilities = np.array([0.33, 0.33, 0.34])
                    predicted_risk = 1
                    confidence = 0.33
                    depth_map = None
            
        # 风险等级映射
        risk_labels = ['低风险', '中风险', '高风险']
        risk_level = risk_labels[predicted_risk]
        
        print(f"风险评估结果: {risk_level} (置信度: {confidence:.4f})")
        return predicted_risk, confidence, probabilities if isinstance(probabilities, np.ndarray) else probabilities.cpu().numpy()[0], depth_map
    
    def make_evasion_decision(self, obstacle_features, motion_features, risk_features):
        """做出避让决策"""
        try:
            decision, confidence = self.evasion_decision_system.make_decision(
                obstacle_features, motion_features, risk_features
            )
        except:
            decision = "继续直行"
            confidence = 0.5
        
        print(f"避让决策结果: {decision} (置信度: {confidence:.4f})")
        return decision, confidence
    
    def process_single_frame(self, image, tracking_sequence=None):
        """处理单帧图像"""
        print("\n=== 开始处理新帧 ===")
        
        # 1. 障碍物检测
        obstacle_class, obstacle_conf, obstacle_probs = self.detect_obstacles(image)
        obstacle_features = obstacle_probs
        
        # 2. 运动预测
        if tracking_sequence is not None and len(tracking_sequence) >= 10:
            motion_class, motion_conf, motion_probs = self.predict_motion(tracking_sequence)
        else:
            motion_probs = np.array([0.33, 0.33, 0.34])
        motion_features = motion_probs
        
        # 3. 深度估计和风险评估
        risk_level, risk_conf, risk_probs, depth_map = self.estimate_depth_and_risk(image)
        risk_features = risk_probs
        
        # 4. 避让决策
        decision, decision_conf = self.make_evasion_decision(obstacle_features, motion_features, risk_features)
        
        print("=== 帧处理完成 ===\n")
        
        # 返回所有结果
        return {
            'obstacle': {
                'class': obstacle_class,
                'confidence': obstacle_conf,
                'probabilities': obstacle_probs
            },
            'motion': {
                'probabilities': motion_probs
            },
            'risk': {
                'level': risk_level,
                'confidence': risk_conf,
                'probabilities': risk_probs
            },
            'decision': {
                'action': decision,
                'confidence': decision_conf
            },
            'depth_map': depth_map
        }


# ==================== 实时深度估计和风险评估功能 ====================
def realtime_depth_estimation_and_risk_assessment():
    """实时深度估计和风险评估功能"""
    print("启动实时深度估计和风险评估系统...")
    
    # 初始化系统
    from depth_estimation import DepthRiskAssessmentSystem
    system = DepthRiskAssessmentSystem()
    system.eval()
    
    # 检查模型文件
    model_path = os.path.join('../models', 'depth_risk_model.pth')
    if os.path.exists(model_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        system.load_state_dict(torch.load(model_path, map_location=device))
        system = system.to(device)
        print("成功加载深度风险评估模型")
    else:
        print("未找到预训练模型，使用随机初始化权重")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("开始实时深度估计和风险评估，按'q'退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 预处理图像
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
        
        # 进行推理
        with torch.no_grad():
            try:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                input_batch = input_batch.to(device)
                risk_scores, depth_map = system(input_batch)
                
                # 计算概率
                probabilities = F.softmax(risk_scores, dim=1)
                
                # 获取风险等级
                risk_level = torch.argmax(probabilities, dim=1).item()
                risk_confidence = torch.max(probabilities).item()
                
                # 风险等级映射
                risk_labels = ['低风险', '中风险', '高风险']
                risk_label = risk_labels[risk_level]
                
                # 在图像上显示结果
                cv2.putText(frame, f'Risk: {risk_label}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Confidence: {risk_confidence:.2f}', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if depth_map is not None:
                    depth_vis = depth_map.cpu().numpy()
                    if len(depth_vis.shape) == 4:
                        depth_vis = depth_vis[0, 0]  # 取第一个批次的第一个通道
                    elif len(depth_vis.shape) == 3:
                        depth_vis = depth_vis[0]  # 取第一个批次
                    
                    # 归一化深度图用于显示
                    depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
                    depth_vis = (depth_vis * 255).astype(np.uint8)
                    
                    # 调整大小以匹配原始帧
                    depth_vis = cv2.resize(depth_vis, (frame.shape[1]//2, frame.shape[0]//2))
                    
                    # 在图像上显示深度图
                    cv2.imshow('Depth Map', depth_vis)
            except Exception as e:
                print(f"推理过程中出错: {e}")
                cv2.putText(frame, 'Error in processing', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow('Real-time Risk Assessment', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("实时深度估计和风险评估系统已关闭")
