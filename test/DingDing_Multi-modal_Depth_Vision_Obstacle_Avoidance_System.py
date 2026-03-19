# DingDing_Multi-modal_Depth_Vision_Obstacle_Avoidance_System.py
import torch, cv2, os, sys, logging, threading, time, warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=UserWarning)

import core.obstacle_detection as obstacle_det
import core.motion_prediction as motion_pred
import core.depth_estimation as depth_est
import core.evasion_decision as evasion_est
from core.core_processing import IntegratedRobotSystem
from core.obstacle_detection import train_obstacle_detection
from core.motion_prediction import train_motion_prediction
from core.depth_estimation import DepthRiskAssessmentSystem, train_depth_risk_model
from core.evasion_decision import train_evasion_model


def get_base_path():
    """获取基础路径"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


# ==================== 日志配置 ====================
def setup_logging():
    """设置日志系统"""
    log_dir = os.path.join(get_base_path(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 生成包含时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"obstacle_avoidance_system_{timestamp}.log")

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger("ObstacleAvoidanceGUI")


# ==================== 现代按钮样式类 ====================
class ModernButton(tk.Button):
    """现代化按钮类，支持悬停和按压效果"""
    def __init__(self, master=None, **kwargs):
        # 默认样式
        self.bg_color = kwargs.pop('bg_color', '#3498db')
        self.fg_color = kwargs.pop('fg_color', 'white')
        self.hover_color = kwargs.pop('hover_color', '#2980b9')
        self.active_color = kwargs.pop('active_color', '#21618c')
        self.disabled_color = kwargs.pop('disabled_color', '#cccccc')  # 禁用颜色
        self.corner_radius = kwargs.pop('corner_radius', 1)
        self.font_size = kwargs.pop('font_size', 10)
        self.padding = kwargs.pop('padding', (4, 2))

        # 设置默认参数
        kwargs.setdefault('bg', self.bg_color)
        kwargs.setdefault('fg', self.fg_color)
        kwargs.setdefault('font', ('微软雅黑', self.font_size, 'bold'))
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('bd', 0)
        kwargs.setdefault('cursor', 'hand2')
        kwargs.setdefault('padx', self.padding[0])
        kwargs.setdefault('pady', self.padding[1])

        super().__init__(master, **kwargs)

        # 保存原始颜色
        self.original_bg = self.bg_color

        # 根据初始状态设置颜色
        if self['state'] == tk.DISABLED:
            self['bg'] = self.disabled_color
        else:
            self['bg'] = self.original_bg

        # 绑定事件
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)

    def on_enter(self, event):
        """鼠标悬停效果"""
        if self['state'] == tk.DISABLED:
            return
        self.config(bg=self.hover_color)

    def on_leave(self, event):
        """鼠标离开效果"""
        if self['state'] == tk.DISABLED:
            return
        self.config(bg=self.original_bg)

    def on_press(self, event):
        """鼠标按下效果"""
        if self['state'] == tk.DISABLED:
            return
        self.config(bg=self.active_color)

    def on_release(self, event):
        """鼠标释放效果"""
        if self['state'] == tk.DISABLED:
            return
        self.config(bg=self.hover_color)

    def config(self, **kwargs):
        """重写config方法，支持状态改变时更新颜色"""
        if 'state' in kwargs:
            state = kwargs['state']
            super().config(**kwargs)
            if state == tk.DISABLED:
                super().config(bg=self.disabled_color)
            else:
                super().config(bg=self.original_bg)
        else:
            super().config(**kwargs)


# ==================== 模型训练工作线程 ====================
class ModelTrainingWorker:
    """模型训练工作线程类"""
    def __init__(self, system, gui):
        self.system = system
        self.gui = gui
        self.stop_requested = False

    def run(self):
        try:
            # 设置全局中断标志
            obstacle_det.training_interrupted = False
            motion_pred.training_interrupted = False
            depth_est.training_interrupted = False
            evasion_est.training_interrupted = False

            if self.system.train_missing_models():
                self.gui.on_model_training_finished(True, "模型训练完成")
            else:
                if (obstacle_det.training_interrupted or
                        motion_pred.training_interrupted or
                        depth_est.training_interrupted or
                        evasion_est.training_interrupted):
                    self.gui.on_model_training_finished(False, "模型训练已被用户中断")
                else:
                    self.gui.on_model_training_finished(False, "没有模型需要训练或训练失败")
        except Exception as e:
            import traceback
            error_msg = f"训练过程中出错: {str(e)}\n{traceback.format_exc()}"
            self.gui.on_model_training_finished(False, error_msg)

    def stop_training(self):
        """请求停止训练"""
        obstacle_det.training_interrupted = True
        motion_pred.training_interrupted = True
        depth_est.training_interrupted = True
        evasion_est.training_interrupted = True


# ==================== 输出重定向器 ====================
class OutputRedirector:
    """输出重定向器，将标准输出重定向到GUI日志显示区域"""
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.last_update = 0
        self.update_interval = 0.1  # 最小更新间隔（秒）

    def write(self, text):
        """重写write方法，将输出发送到GUI"""
        if text.strip():
            current_time = time.time()
            # 限制日志更新频率
            if current_time - self.last_update > self.update_interval:
                try:
                    self.gui.log_message(text.strip(), level="info")
                    self.last_update = current_time
                except:
                    pass
        self.stdout.write(text)

    def flush(self):
        """flush方法"""
        self.stdout.flush()

    def isatty(self):
        """isatty方法"""
        return False


# ==================== 训练线程 ====================
class TrainingThread(threading.Thread):
    """训练线程类，用于在后台执行模型训练"""
    def __init__(self, model_type, gui):
        super().__init__()
        self.model_type = model_type
        self.gui = gui
        self.stop_requested = False  # 添加停止请求标志
        self.daemon = True  # 设置为守护线程

    def run(self):
        """执行训练任务"""
        try:
            self.gui.on_training_progress(f"开始训练 {self.model_type} 模型...")

            # 根据模型类型调用相应的训练函数
            if self.model_type == "obstacle":
                self._train_with_interruption_support(train_obstacle_detection)
            elif self.model_type == "motion":
                self._train_with_interruption_support(train_motion_prediction)
            elif self.model_type == "depth":
                self._train_with_interruption_support(train_depth_risk_model)
            elif self.model_type == "evasion":
                self._train_with_interruption_support(self._train_evasion_model)
            else:
                raise ValueError(f"未知的模型类型: {self.model_type}")

            if self.stop_requested:
                self.gui.on_training_finished(f"{self.model_type} 模型训练已停止")
            else:
                self.gui.on_training_finished(f"{self.model_type} 模型训练完成")
        except Exception as e:
            if self.stop_requested:
                self.gui.on_training_finished(f"{self.model_type} 模型训练已停止")
            else:
                error_msg = f"{self.model_type} 模型训练出错: {str(e)}"
                logger.error(error_msg)
                self.gui.on_training_error(error_msg)

    def _train_with_interruption_support(self, train_function):
        """带中断支持的训练函数包装器"""
        # 根据模型类型设置相应的中断标志
        if self.model_type == "obstacle":
            obstacle_det.training_interrupted = False
        elif self.model_type == "motion":
            motion_pred.training_interrupted = False
        elif self.model_type == "depth":
            depth_est.training_interrupted = False
        elif self.model_type == "evasion":
            evasion_est.training_interrupted = False

        # 调用训练函数并传递进度回调
        if self.model_type == "obstacle":
            train_function(self._progress_callback)
        elif self.model_type == "motion":
            train_function(progress_callback=self._progress_callback)
        elif self.model_type == "depth":
            train_function(progress_callback=self._progress_callback)
        elif self.model_type == "evasion":
            self._train_evasion_model()

    def _train_evasion_model(self):
        """训练避让决策模型"""
        try:
            train_evasion_model(progress_callback=self._progress_callback)
        except Exception as e:
            print(f"避让决策模型训练出错: {str(e)}")

    def _progress_callback(self, progress, message):
        """进度回调函数"""
        # 更新进度条
        self.gui.root.after(0, lambda: self.gui.progress_var.set(progress))
        # 更新进度标签
        self.gui.root.after(0, lambda: self.gui.progress_label.config(text=f"{progress}%"))
        # 更新状态栏
        self.gui.root.after(0, lambda: self.gui.status_bar.config(text=message))
        # 记录日志
        self.gui.root.after(0, lambda: self.gui.log_message(message))

    def _update_progress_label_position(self, event=None):
        """更新进度标签位置"""
        # 确保标签在进度条中央
        self.progress_label.place_configure(relx=0.5, rely=0.5, anchor="center")

    def stop_training(self):
        """请求停止训练"""
        self.stop_requested = True
        self.gui.on_training_progress("正在停止训练...")

        # 根据模型类型设置相应的中断标志
        if self.model_type == "obstacle":
            obstacle_det.training_interrupted = True
        elif self.model_type == "motion":
            motion_pred.training_interrupted = True
        elif self.model_type == "depth":
            depth_est.training_interrupted = True
        elif self.model_type == "evasion":
            evasion_est.training_interrupted = True


# ==================== 全部重训练线程 ====================
class RetrainAllThread(threading.Thread):
    """全部重训练线程类，用于重新训练所有模型"""
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.stop_requested = False
        self.daemon = True

    def run(self):
        """执行全部重训练任务"""
        try:
            self.gui.on_training_progress("开始全部重新训练...")

            # 依次训练所有模型
            models = ["obstacle", "motion", "depth", "evasion"]
            model_names = ["障碍物检测", "运动预测", "深度风险", "避让决策"]

            for i, (model_type, model_name) in enumerate(zip(models, model_names)):
                if self.stop_requested:
                    break

                self.gui.on_training_progress(f"开始训练 {model_name} 模型 ({i + 1}/4)...")

                # 根据模型类型调用相应的训练函数
                if model_type == "obstacle":
                    self._train_with_interruption_support(train_obstacle_detection, model_type)
                elif model_type == "motion":
                    self._train_with_interruption_support(train_motion_prediction, model_type)
                elif model_type == "depth":
                    self._train_with_interruption_support(train_depth_risk_model, model_type)
                elif model_type == "evasion":
                    self._train_with_interruption_support(self._train_evasion_model, model_type)

                if self.stop_requested:
                    self.gui.on_training_finished(f"{model_name} 模型训练已停止")
                else:
                    self.gui.on_training_finished(f"{model_name} 模型训练完成")

            if self.stop_requested:
                self.gui.on_training_finished("全部重新训练已停止")
            else:
                self.gui.on_training_finished("全部重新训练完成")

        except Exception as e:
            if self.stop_requested:
                self.gui.on_training_finished("全部重新训练已停止")
            else:
                error_msg = f"全部重新训练出错: {str(e)}"
                logger.error(error_msg)
                self.gui.on_training_error(error_msg)

    def _train_with_interruption_support(self, train_function, model_type):
        """带中断支持的训练函数包装器"""
        # 设置相应的中断标志
        if model_type == "obstacle":
            obstacle_det.training_interrupted = False
        elif model_type == "motion":
            motion_pred.training_interrupted = False
        elif model_type == "depth":
            depth_est.training_interrupted = False
        elif model_type == "evasion":
            evasion_est.training_interrupted = False

        # 调用训练函数并传递进度回调
        if model_type == "obstacle":
            train_function(self._progress_callback)
        elif model_type == "motion":
            train_function(progress_callback=self._progress_callback)
        elif model_type == "depth":
            train_function(progress_callback=self._progress_callback)
        elif model_type == "evasion":
            self._train_evasion_model()

    def _train_evasion_model(self):
        """训练避让决策模型"""
        try:
            train_evasion_model(progress_callback=self._progress_callback)
        except Exception as e:
            print(f"避让决策模型训练出错: {str(e)}")

    def _progress_callback(self, progress, message):
        """进度回调函数"""
        # 更新进度条
        self.gui.root.after(0, lambda: self.gui.progress_var.set(progress))
        # 更新状态栏
        self.gui.root.after(0, lambda: self.gui.status_bar.config(text=message))
        # 记录日志
        self.gui.root.after(0, lambda: self.gui.log_message(message))

    def stop_training(self):
        """请求停止训练"""
        self.stop_requested = True
        self.gui.on_training_progress("正在停止全部训练...")

        # 设置所有中断标志
        obstacle_det.training_interrupted = True
        motion_pred.training_interrupted = True
        depth_est.training_interrupted = True
        evasion_est.training_interrupted = True


# ==================== 摄像头线程 ====================
class CameraThread(threading.Thread):
    """摄像头线程类"""
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.running = False
        self.cap = None
        self.daemon = True

    def run(self):
        """运行摄像头捕获循环"""
        self.running = True
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.gui.on_camera_error("无法打开摄像头")
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.gui.display_camera_frame(frame)
            else:
                self.gui.on_camera_error("无法读取摄像头帧")
                break
            time.sleep(0.01)  # 控制帧率

    def stop(self):
        """停止摄像头捕获"""
        self.running = False
        if self.cap:
            self.cap.release()


# ==================== 实时深度估计线程 ====================
class RealTimeDepthThread(threading.Thread):
    """实时深度估计线程类"""
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        self.running = False
        self.system = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.daemon = True

    def run(self):
        """运行实时深度估计"""
        self.running = True

        # 初始化系统
        try:
            self.system = DepthRiskAssessmentSystem()
            self.system.eval()

            # 检查模型文件
            model_path = os.path.join(get_base_path(), 'models', 'depth_risk_model.pth')
            if os.path.exists(model_path):
                self.system.load_state_dict(torch.load(model_path, map_location=self.device))
                self.system = self.system.to(self.device)
                print("成功加载深度风险评估模型")
            else:
                print("未找到预训练模型，使用随机初始化权重")
        except Exception as e:
            self.gui.on_depth_error(f"初始化系统失败: {str(e)}")
            return

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.gui.on_depth_error("无法打开摄像头")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.gui.on_depth_error("无法读取摄像头帧")
                break

            try:
                # 转换颜色空间
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # 预处理图像
                input_tensor = self.transform(pil_image)
                input_batch = input_tensor.unsqueeze(0).to(self.device)

                # 进行推理
                with torch.no_grad():
                    risk_scores, depth_map = self.system(input_batch)

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
                            depth_vis = depth_vis[0, 0]
                        elif len(depth_vis.shape) == 3:
                            depth_vis = depth_vis[0]

                        # 归一化深度图用于显示
                        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
                        depth_vis = (depth_vis * 255).astype(np.uint8)

                        # 调整大小以匹配原始帧
                        depth_vis = cv2.resize(depth_vis, (frame.shape[1] // 2, frame.shape[0] // 2))

                        # 应用伪彩色映射
                        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # 发送信号
                self.gui.display_depth_frame(frame, risk_label, risk_confidence)

            except Exception as e:
                self.gui.on_depth_error(f"处理帧时出错: {str(e)}")

            time.sleep(0.01)

        # 释放资源
        cap.release()

    def stop(self):
        """停止实时深度估计"""
        self.running = False


# ==================== 主窗口 ====================
class ObstacleAvoidanceGUI:
    """叮丁多模态深度视觉避障系统主窗口"""
    def __init__(self, root):
        self.root = root
        self.root.title("叮丁多模态深度视觉避障系统")
        self.root.geometry("1200x800")
        self.root.state('zoomed')

        # 设置主题颜色
        self.set_theme()

        # 初始化变量
        self.camera_thread = None
        self.training_thread = None
        self.depth_thread = None
        self.system = None
        self.current_image_path = None

        # 初始化输出重定向
        self.output_redirector = OutputRedirector(self)
        sys.stdout = self.output_redirector
        sys.stderr = self.output_redirector

        # 创建界面
        self.create_menu_bar()
        self.create_widgets()

        # 初始化系统
        self.init_system()
        self.root.iconbitmap(os.path.join(get_base_path(), "img", "icon.ico"))

        logger.info("叮丁多模态深度视觉避障系统GUI启动")

    def set_theme(self):
        """设置主题颜色"""
        # 主色调
        self.primary_color = "#2c3e50"  # 深蓝色
        self.secondary_color = "#34495e"  # 次深蓝
        self.accent_color = "#3498db"  # 蓝色
        self.success_color = "#27ae60"  # 绿色
        self.warning_color = "#f39c12"  # 橙色
        self.danger_color = "#e74c3c"  # 红色
        self.info_color = "#17a2b8"  # 青色

        # 背景色
        self.bg_color = "#ecf0f1"  # 浅灰
        self.card_bg = "#ffffff"  # 白色
        self.border_color = "#bdc3c7"  # 边框灰

        # 文字颜色
        self.text_primary = "#2c3e50"
        self.text_secondary = "#7f8c8d"
        self.text_light = "#ecf0f1"

        # 按钮颜色
        self.btn_primary = "#3498db"
        self.btn_primary_hover = "#2980b9"
        self.btn_primary_active = "#21618c"


        self.btn_success = "#27ae60"
        self.btn_success_hover = "#219653"
        self.btn_success_active = "#1e8449"

        self.btn_warning = "#f39c12"
        self.btn_warning_hover = "#e67e22"
        self.btn_warning_active = "#d35400"

        self.btn_danger = "#e74c3c"
        self.btn_danger_hover = "#c0392b"
        self.btn_danger_active = "#a93226"

        self.btn_info = "#17a2b8"
        self.btn_info_hover = "#138496"
        self.btn_info_active = "#117a8b"

        # 设置窗口背景
        self.root.configure(bg=self.bg_color)

    def create_menu_bar(self):
        """创建菜单栏"""
        self.menubar = tk.Menu(self.root, bg=self.primary_color, fg=self.text_light, activeborderwidth=5,
                          activebackground=self.accent_color, activeforeground=self.text_light,
                          font=('微软雅黑', 12), relief=tk.FLAT, bd=6)
        self.root.config(menu=self.menubar)

        # 文件菜单
        self.file_menu = tk.Menu(self.menubar, tearoff=0, bg=self.secondary_color, fg=self.text_light,
                            activebackground=self.accent_color, activeforeground=self.text_light,
                            font=('微软雅黑', 12))
        self.menubar.add_cascade(label="  文件  ", menu=self.file_menu, font=('微软雅黑', 10, 'bold'))
        self.file_menu.add_command(label="加载图像", command=self.load_image, font=('微软雅黑', 10))
        self.file_menu.add_command(label="保存结果", command=self.save_results, font=('微软雅黑', 10))
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.root.quit, font=('微软雅黑', 10))

        # 训练菜单
        self.train_menu = tk.Menu(self.menubar, tearoff=0, bg=self.secondary_color, fg=self.text_light,
                             activebackground=self.accent_color, activeforeground=self.text_light,
                             font=('微软雅黑', 12))
        self.menubar.add_cascade(label="  训练  ", menu=self.train_menu, font=('微软雅黑', 12, 'bold'))
        self.train_menu.add_command(label="全部重新训练", command=lambda: self.retrain_all_models(), font=('微软雅黑', 10))
        self.train_menu.add_command(label="训练障碍物检测模型", command=lambda: self.train_model("obstacle"), font=('微软雅黑', 10))
        self.train_menu.add_command(label="训练运动预测模型", command=lambda: self.train_model("motion"), font=('微软雅黑', 10))
        self.train_menu.add_command(label="训练深度风险模型", command=lambda: self.train_model("depth"), font=('微软雅黑', 10))
        self.train_menu.add_command(label="训练避让决策模型", command=lambda: self.train_model("evasion"), font=('微软雅黑', 10))

        # 视图菜单
        self.view_menu = tk.Menu(self.menubar, tearoff=0, bg=self.secondary_color, fg=self.text_light,
                            activebackground=self.accent_color, activeforeground=self.text_light,
                            font=('微软雅黑', 12))
        self.menubar.add_cascade(label="  视图  ", menu=self.view_menu, font=('微软雅黑', 12, 'bold'))
        self.view_menu.add_command(label="开启摄像头", command=self.toggle_camera, font=('微软雅黑', 10))
        self.view_menu.add_command(label="实时深度估计", command=self.toggle_realtime_depth, font=('微软雅黑', 10))

        # 帮助菜单
        self.help_menu = tk.Menu(self.menubar, tearoff=0, bg=self.secondary_color, fg=self.text_light,
                            activebackground=self.accent_color, activeforeground=self.text_light,
                            font=('微软雅黑', 12))
        self.menubar.add_cascade(label="  帮助  ", menu=self.help_menu, font=('微软雅黑', 12, 'bold'))
        self.help_menu.add_command(label="关于", command=self.show_about, font=('微软雅黑', 10))

    def create_widgets(self):
        """创建主界面组件"""
        # 创建状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建左右两个面板
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 左侧控制面板
        self.create_left_panel(left_panel)

        # 右侧显示面板
        self.create_right_panel(right_panel)

    def create_left_panel(self, parent):
        """创建左侧控制面板"""
        # 控制组
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 图像处理区域
        image_frame = ttk.LabelFrame(control_frame, text="图像处理", padding=10)
        image_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建两列按钮布局
        image_buttons_frame = tk.Frame(image_frame, bg=self.card_bg)
        image_buttons_frame.pack(fill=tk.X)

        # 第一列
        col1 = tk.Frame(image_buttons_frame, bg=self.card_bg)
        col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.load_btn = ModernButton(
            col1,
            text="加载测试图像",
            command=self.load_image,
            bg_color=self.btn_primary,
            hover_color=self.btn_primary_hover,
            active_color=self.btn_primary_active
        )
        self.load_btn.pack(fill=tk.X, pady=2)

        # 第二列
        col2 = tk.Frame(image_buttons_frame, bg=self.card_bg)
        col2.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        self.process_btn = ModernButton(
            col2,
            text="处理图像",
            command=self.process_image_and_switch_tab,
            state=tk.DISABLED,
            bg_color=self.btn_success,
            hover_color=self.btn_success_hover,
            active_color=self.btn_success_active
        )
        self.process_btn.pack(fill=tk.X, pady=2)

        # 模型训练区域
        train_frame = ttk.LabelFrame(control_frame, text="模型训练", padding=10)
        train_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建两列训练按钮
        train_buttons_frame = tk.Frame(train_frame, bg=self.card_bg)
        train_buttons_frame.pack(fill=tk.X)

        # 第一列训练按钮
        train_col1 = tk.Frame(train_buttons_frame, bg=self.card_bg)
        train_col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.train_obstacle_btn = ModernButton(
            train_col1,
            text="训练障碍物检测",
            command=self.train_obstacle_and_switch_tab,
            bg_color=self.btn_success,
            hover_color=self.btn_success_hover,
            active_color=self.btn_success_active
        )
        self.train_obstacle_btn.pack(fill=tk.X, pady=2)

        self.train_motion_btn = ModernButton(
            train_col1,
            text="训练运动预测",
            command=self.train_motion_and_switch_tab,
            bg_color=self.btn_success,
            hover_color=self.btn_success_hover,
            active_color=self.btn_success_active
        )
        self.train_motion_btn.pack(fill=tk.X, pady=2)

        self.retrain_all_btn = ModernButton(
            train_col1,
            text="全部重新训练",
            command=self.retrain_all_models_and_switch_tab,
            bg_color=self.btn_warning,
            hover_color=self.btn_warning_hover,
            active_color=self.btn_warning_active
        )
        self.retrain_all_btn.pack(fill=tk.X, pady=2)

        # 第二列训练按钮
        train_col2 = tk.Frame(train_buttons_frame, bg=self.card_bg)
        train_col2.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        self.train_depth_btn = ModernButton(
            train_col2,
            text="训练深度风险",
            command=self.train_depth_and_switch_tab,
            bg_color=self.btn_success,
            hover_color=self.btn_success_hover,
            active_color=self.btn_success_active
        )
        self.train_depth_btn.pack(fill=tk.X, pady=2)

        self.train_evasion_btn = ModernButton(
            train_col2,
            text="训练避让决策",
            command=self.train_evasion_and_switch_tab,
            bg_color=self.btn_success,
            hover_color=self.btn_success_hover,
            active_color=self.btn_success_active
        )
        self.train_evasion_btn.pack(fill=tk.X, pady=2)

        self.stop_training_btn = ModernButton(
            train_col2,
            text="停止训练",
            command=self.stop_training,
            state=tk.DISABLED,
            bg_color=self.btn_danger,
            hover_color=self.btn_danger_hover,
            active_color=self.btn_danger_active
        )
        self.stop_training_btn.pack(fill=tk.X, pady=2)

        # 实时处理区域
        realtime_frame = ttk.LabelFrame(control_frame, text="实时处理", padding=10)
        realtime_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建两列实时处理按钮
        realtime_buttons_frame = tk.Frame(realtime_frame, bg=self.card_bg)
        realtime_buttons_frame.pack(fill=tk.X)

        # 第一列实时按钮
        realtime_col1 = tk.Frame(realtime_buttons_frame, bg=self.card_bg)
        realtime_col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # 第二列实时按钮
        realtime_col2 = tk.Frame(realtime_buttons_frame, bg=self.card_bg)
        realtime_col2.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        self.camera_btn = ModernButton(
            realtime_col1,
            text="开启摄像头",
            command=self.toggle_camera_and_switch_tab,
            bg_color=self.btn_info,
            hover_color=self.btn_info_hover,
            active_color=self.btn_info_active
        )
        self.camera_btn.pack(fill=tk.X, pady=2)

        self.depth_btn = ModernButton(
            realtime_col2,
            text="实时深度估计",
            command=self.toggle_realtime_depth_and_switch_tab,
            bg_color=self.btn_info,
            hover_color=self.btn_info_hover,
            active_color=self.btn_info_active
        )
        self.depth_btn.pack(fill=tk.X, pady=2)

        # 模型选择
        model_frame = ttk.LabelFrame(control_frame, text="模型选择", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        # 障碍物检测模型
        obstacle_model_frame = ttk.Frame(model_frame)
        obstacle_model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(obstacle_model_frame, text="障碍物检测模型:", width=13).pack(side=tk.LEFT)
        self.obstacle_model_var = tk.StringVar(value="CNN-SwinTransformer")
        ttk.Combobox(obstacle_model_frame, textvariable=self.obstacle_model_var,
                     values=["CNN-SwinTransformer"], state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # 运动预测模型
        motion_model_frame = ttk.Frame(model_frame)
        motion_model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(motion_model_frame, text="运动预测模型: ", width=13).pack(side=tk.LEFT)
        self.motion_model_var = tk.StringVar(value="LSTM-Transformer")
        ttk.Combobox(motion_model_frame, textvariable=self.motion_model_var,
                     values=["LSTM-Transformer"], state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # 深度估计模型
        depth_model_frame = ttk.Frame(model_frame)
        depth_model_frame.pack(fill=tk.X, pady=2)
        ttk.Label(depth_model_frame, text="深度估计模型: ", width=13).pack(side=tk.LEFT)
        self.depth_model_var = tk.StringVar(value="MiDaS-ResNet")
        ttk.Combobox(depth_model_frame, textvariable=self.depth_model_var,
                     values=["MiDaS-ResNet"], state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # 参数设置
        param_frame = ttk.LabelFrame(control_frame, text="参数设置", padding=10)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # 置信度阈值
        conf_frame = ttk.Frame(param_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(conf_frame, text="置信度阈值:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.5)
        conf_spin = ttk.Spinbox(conf_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.confidence_var,
                                width=10)
        conf_spin.pack(side=tk.RIGHT)

        # 处理延迟
        delay_frame = ttk.Frame(param_frame)
        delay_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(delay_frame, text="处理延迟(毫秒):").pack(side=tk.LEFT)
        self.delay_var = tk.IntVar(value=1000)
        delay_spin = ttk.Spinbox(delay_frame, from_=0, to=5000, increment=100, textvariable=self.delay_var, width=10)
        delay_spin.pack(side=tk.RIGHT)

        # 进度条
        self.progress_var = tk.IntVar()
        progress_frame = tk.Frame(control_frame, bg=self.card_bg)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 进度标签
        self.progress_label = tk.Label(progress_frame, text="0%", font=('微软雅黑', 9), fg=self.text_primary, bg=self.card_bg)
        self.progress_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # 日志显示
        log_frame = ttk.LabelFrame(parent, text="系统日志", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_display = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            state=tk.DISABLED,
            bg="#f8f9fa",
            fg=self.text_primary,
            font=('Consolas', 9)
        )
        self.log_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ModernButton(
            log_frame,
            text="清空日志",
            command=self.clear_log,
            bg_color=self.btn_primary,
            hover_color=self.btn_primary_hover,
            active_color=self.btn_primary_active,
            padding=(8, 4)
        ).pack(pady=5)

    def create_right_panel(self, parent):
        """创建右侧显示面板"""
        # 创建笔记本控件用于标签页
        style = ttk.Style()
        style.configure("TNotebook", background=self.bg_color)
        style.configure("TNotebook.Tab", font=('微软雅黑', 10, 'bold'), padding=[10, 5])

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 原始图像标签页
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="原始图像")
        self.original_label = ttk.Label(self.original_frame, text="原始图像将显示在这里", background=self.card_bg)
        self.original_label.pack(expand=True)
        self.original_label.configure(anchor="center")

        # 处理结果标签页
        self.result_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.result_frame, text="处理结果")
        self.result_label = ttk.Label(self.result_frame, text="处理结果将显示在这里", background=self.card_bg)
        self.result_label.pack(expand=True)
        self.result_label.configure(anchor="center")

        # 深度图标签页
        self.depth_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.depth_frame, text="深度图")
        self.depth_label = ttk.Label(self.depth_frame, text="深度图将显示在这里", background=self.card_bg)
        self.depth_label.pack(expand=True)
        self.depth_label.configure(anchor="center")

        # 摄像头画面标签页
        self.camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_frame, text="摄像头画面")
        self.camera_label = ttk.Label(self.camera_frame, text="摄像头画面将显示在这里", background=self.card_bg)
        self.camera_label.pack(expand=True)
        self.camera_label.configure(anchor="center")

        # 实时深度估计标签页
        self.depth_realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.depth_realtime_frame, text="实时深度估计")
        self.depth_realtime_label = ttk.Label(self.depth_realtime_frame, text="实时深度估计将显示在这里",
                                              background=self.card_bg)
        self.depth_realtime_label.pack(expand=True)
        self.depth_realtime_label.configure(anchor="center")

        # 结果信息显示
        result_frame = ttk.LabelFrame(parent, text="检测结果")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.result_display = scrolledtext.ScrolledText(
            result_frame,
            height=8,
            state=tk.DISABLED,
            bg="#f8f9fa",
            fg=self.text_primary,
            font=('Consolas', 9)
        )
        self.result_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def init_system(self):
        """初始化系统"""
        try:
            self.system = IntegratedRobotSystem()
            self.log_message("系统初始化完成")
            self.status_bar.config(text="系统就绪")

            if hasattr(self.system, 'models_to_train') and self.system.models_to_train:
                self.root.after(1000, self.prompt_for_missing_models)
        except Exception as e:
            error_msg = f"系统初始化失败: {str(e)}"
            self.log_message(error_msg, level="error")
            self.status_bar.config(text="系统初始化失败")
            logger.error(error_msg)

    def prompt_for_missing_models(self):
        """提示用户训练缺失的模型"""
        # 创建一个队列来存储需要训练的模型
        self.models_to_prompt = self.system.models_to_train.copy()
        self.current_prompt_index = 0

        # 开始逐一提示
        self.prompt_next_model()

    def prompt_next_model(self):
        """逐一提示训练模型"""
        # 如果还有模型需要提示
        if self.current_prompt_index < len(self.models_to_prompt):
            model_type = self.models_to_prompt[self.current_prompt_index]

            # 获取模型名称
            model_names = {
                "obstacle": "障碍物检测",
                "motion": "运动预测",
                "depth": "深度风险评估",
                "evasion": "避让决策"
            }
            model_name = model_names.get(model_type, model_type)

            # 提示用户是否训练该模型
            reply = messagebox.askyesno(
                '模型缺失',
                f'检测到缺失的模型: {model_name}\n是否现在开始训练该模型?'
            )

            if reply:
                self.train_model(model_type)
            else:
                self.current_prompt_index += 1
                self.prompt_next_model()
        else:
            self.log_message("缺失模型提示完成")

    def log_message(self, message, level="info"):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 根据日志级别设置颜色
        color_tags = {
            "info": "black",
            "warning": "orange",
            "error": "red"
        }
        color = color_tags.get(level, "black")

        log_entry = f"[{timestamp}] {message}\n"

        # 添加到日志显示区域
        self.log_display.config(state=tk.NORMAL)
        self.log_display.insert(tk.END, log_entry)
        self.log_display.tag_configure(level, foreground=color)
        self.log_display.config(state=tk.DISABLED)
        self.log_display.see(tk.END)  # 自动滚动到底部

        # 记录到文件
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)

    def retrain_all_models(self):
        """全部重新训练功能"""
        reply = messagebox.askyesno(
            '确认重训练',
            '确定要重新训练所有模型吗？\n注意：这将覆盖现有模型文件，且可能需要较长时间。'
        )

        if reply:
            self.set_all_controls_enabled(False)

            # 创建并启动全部重训练线程
            self.training_thread = RetrainAllThread(self)
            self.training_thread.start()

            self.log_message("开始全部重新训练...")
            self.status_bar.config(text="正在全部重新训练...")

    def train_model(self, model_type):
        """训练指定模型"""
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("警告", "已有模型正在训练中，请等待完成后再开始新的训练。")
            return

        # 禁用所有控件，启用停止训练按钮
        self.set_all_controls_enabled(False)

        # 创建并启动训练线程
        if model_type == "all":
            self.training_thread = RetrainAllThread(self)
        else:
            self.training_thread = TrainingThread(model_type, self)
        self.training_thread.start()

        self.log_message(f"开始训练 {model_type} 模型...")
        self.status_bar.config(text=f"正在训练 {model_type} 模型...")
        # 重置进度显示
        self.progress_var.set(0)
        self.progress_label.config(text="0%")

    def stop_training(self):
        """停止当前训练"""
        if self.training_thread and self.training_thread.is_alive():
            reply = messagebox.askyesno(
                '确认停止',
                '确定要停止当前训练吗？\n注意：这可能会导致训练进度丢失。'
            )

            if reply:
                self.log_message("正在请求停止训练...")
                self.status_bar.config(text="正在停止训练...")
                self.training_thread.stop_training()

    def on_training_finished(self, message):
        """训练完成回调"""
        self.log_message(message)
        self.status_bar.config(text="训练完成" if "完成" in message else "训练已停止")

        # 启用所有控件，禁用停止训练按钮
        self.set_all_controls_enabled(True)

        self.progress_var.set(0)  # 重置进度条
        messagebox.showinfo("训练完成", message)

        # 继续提示下一个模型
        if hasattr(self, 'models_to_prompt') and hasattr(self, 'current_prompt_index'):
            self.current_prompt_index += 1
            self.prompt_next_model()

    def on_model_training_finished(self, success, message):
        """模型训练完成回调"""
        if success:
            self.log_message("模型训练完成")
            messagebox.showinfo("训练完成", "模型训练已完成，系统功能现已完整可用。")
        else:
            self.log_message(message, level="error")
            messagebox.showerror("训练出错", message)

        # 启用所有控件，禁用停止训练按钮
        self.set_all_controls_enabled(True)

        self.progress_var.set(0)
        self.status_bar.config(text="就绪")

    def on_training_progress(self, message):
        """训练进度回调"""
        self.log_message(message)
        self.status_bar.config(text=message)

    def on_training_error(self, error_message):
        """训练错误回调"""
        self.log_message(error_message, level="error")
        self.status_bar.config(text="训练出错")

        # 启用所有控件，禁用停止训练按钮
        self.set_all_controls_enabled(True)

        messagebox.showerror("训练出错", error_message)

    def set_all_controls_enabled(self, enabled):
        """设置所有控件的启用状态"""
        state = tk.NORMAL if enabled else tk.DISABLED

        # 禁用/启用左侧控制面板的所有按钮
        self.load_btn.config(state=state)
        self.process_btn.config(state=state if self.current_image_path else tk.DISABLED)

        # 禁用/启用训练按钮
        self.train_obstacle_btn.config(state=state)
        self.train_motion_btn.config(state=state)
        self.train_depth_btn.config(state=state)
        self.train_evasion_btn.config(state=state)
        self.retrain_all_btn.config(state=state)

        # 特别处理停止训练按钮状态
        if not enabled:
            self.stop_training_btn.config(state=tk.NORMAL)
        else:
            self.stop_training_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
            self.progress_label.config(text="0%")

        # 禁用/启用实时处理按钮
        self.camera_btn.config(state=state)
        self.depth_btn.config(state=state)

        # 禁用/启用菜单项
        self._set_menu_state(enabled)

    def _set_menu_state(self, enabled):
        """设置菜单项的启用状态"""
        state = tk.NORMAL if enabled else tk.DISABLED

        # 文件菜单项
        self.file_menu.entryconfig("加载图像", state=state)
        self.file_menu.entryconfig("保存结果", state=state)

        # 训练菜单项
        self.train_menu.entryconfig("全部重新训练", state=state)
        self.train_menu.entryconfig("训练障碍物检测模型", state=state)
        self.train_menu.entryconfig("训练运动预测模型", state=state)
        self.train_menu.entryconfig("训练深度风险模型", state=state)
        self.train_menu.entryconfig("训练避让决策模型", state=state)

        # 视图菜单项
        self.view_menu.entryconfig("开启摄像头", state=state)
        self.view_menu.entryconfig("实时深度估计", state=state)

    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )

        if file_path:
            try:
                self.current_image_path = file_path

                # 显示原始图像
                image = Image.open(file_path)

                # 使用新的转换方法显示图像
                photo = self.convert_cv_to_photo(np.array(image))
                
                # 保存引用以防止被垃圾回收
                self.original_image_photo = photo
                self.original_label.config(image=photo, text="")

                self.process_btn.config(state=tk.NORMAL)
                self.log_message(f"已加载图像: {file_path}")
                self.status_bar.config(text="图像加载完成")
            except Exception as e:
                error_msg = f"加载图像失败: {str(e)}"
                self.log_message(error_msg, level="error")
                messagebox.showerror("错误", error_msg)

    def process_image(self):
        """处理图像"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先加载图像")
            return

        if not self.system:
            messagebox.showerror("错误", "系统未初始化")
            return

        try:
            self.status_bar.config(text="正在处理图像...")
            self.progress_var.set(30)

            # 读取图像
            image = cv2.imread(self.current_image_path)
            
            # 检查图像是否成功加载
            if image is None:
                # 尝试使用PIL加载图像然后转换为OpenCV格式
                try:
                    pil_image = Image.open(self.current_image_path)
                    # 转换PIL图像为OpenCV格式
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    self.log_message("使用备用方法成功加载图像", level="info")
                except Exception as e:
                    error_msg = f"无法加载图像文件: {self.current_image_path}，请检查文件路径和文件完整性。错误详情: {str(e)}"
                    self.log_message(error_msg, level="error")
                    self.status_bar.config(text="处理图像出错")
                    self.progress_var.set(0)
                    messagebox.showerror("错误", error_msg)
                    return

            # 处理图像
            results = self.system.process_single_frame(image)

            self.progress_var.set(70)

            # 显示结果
            self.display_results(results, image)

            self.progress_var.set(100)
            self.status_bar.config(text="图像处理完成")
            self.log_message("图像处理完成")

            # 重置进度条
            self.root.after(2000, lambda: self.progress_var.set(0))

        except Exception as e:
            error_msg = f"处理图像时出错: {str(e)}"
            self.log_message(error_msg, level="error")
            self.status_bar.config(text="处理图像出错")
            self.progress_var.set(0)
            messagebox.showerror("错误", error_msg)

    def display_results(self, results, original_image):
        """显示处理结果"""
        result_image = original_image.copy()

        # 在图像上绘制检测结果
        obstacle_class = results['obstacle']['class'] if results['obstacle'] is not None else 0
        obstacle_conf = results['obstacle']['confidence'] if results['obstacle'] is not None else 0.0
        risk_level = results['risk']['level'] if results['risk'] is not None else "未知"
        risk_conf = results['risk']['confidence'] if results['risk'] is not None else 0.0
        decision = results['decision']['action'] if results['decision'] is not None else "未知"
        decision_conf = results['decision']['confidence'] if results['decision'] is not None else 0.0

        # 绘制文本
        cv2.putText(result_image, f'Obstacle: {["无障碍物", "行人", "车辆"][obstacle_class]} ({obstacle_conf:.2f})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_image, f'Risk: {risk_level} ({risk_conf:.2f})',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_image, f'Decision: {decision} ({decision_conf:.2f})',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 转换为PhotoImage并显示
        result_photo = self.convert_cv_to_photo(result_image)
        self.result_image_photo = result_photo
        self.result_label.config(image=result_photo, text="")

        # 显示深度图
        if results.get('depth_map') is not None:
            try:
                depth_map = results['depth_map'].cpu().numpy()
                if len(depth_map.shape) == 4:
                    depth_vis = depth_map[0, 0]
                elif len(depth_map.shape) == 3:
                    depth_vis = depth_map[0]
                else:
                    depth_vis = depth_map

                # 归一化深度图用于显示
                depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
                depth_vis = (depth_vis * 255).astype(np.uint8)

                # 转换为伪彩色图像
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # 转换为PhotoImage并显示
                depth_photo = self.convert_cv_to_photo(depth_colormap)
                self.depth_image_photo = depth_photo  # 保存引用
                self.depth_label.config(image=depth_photo, text="")
            except Exception as e:
                self.log_message(f"显示深度图时出错: {str(e)}", level="warning")

        # 显示文本结果
        result_text = f"""处理结果摘要:
障碍物类型: {['无障碍物', '行人', '车辆'][obstacle_class]}
障碍物置信度: {obstacle_conf:.4f}
风险评估等级: {risk_level}
风险评估置信度: {risk_conf:.4f}
建议行动: {decision}
行动置信度: {decision_conf:.4f}

详细概率分布:
障碍物概率分布: {results['obstacle'].get('probabilities', 'N/A') if results['obstacle'] is not None else 'N/A'}
运动预测概率分布: {results['motion'].get('probabilities', 'N/A') if results['motion'] is not None else 'N/A'}
风险评估概率分布: {results['risk'].get('probabilities', 'N/A') if results['risk'] is not None else 'N/A'}
"""
        self.result_display.config(state=tk.NORMAL)
        self.result_display.delete(1.0, tk.END)
        self.result_display.insert(tk.END, result_text)
        self.result_display.config(state=tk.DISABLED)

    def update_result_display(self, results):
        """更新结果显示区域"""
        if results is None:
            self.log_message("无法更新空的结果显示", level="warning")
            return
            
        obstacle_class = results['obstacle']['class'] if results['obstacle'] is not None else 0
        obstacle_conf = results['obstacle']['confidence'] if results['obstacle'] is not None else 0.0
        risk_level = results['risk']['level'] if results['risk'] is not None else "未知"
        risk_conf = results['risk']['confidence'] if results['risk'] is not None else 0.0
        decision = results['decision']['action'] if results['decision'] is not None else "未知"
        decision_conf = results['decision']['confidence'] if results['decision'] is not None else 0.0

        result_text = f"""处理结果摘要:
障碍物类型: {['无障碍物', '行人', '车辆'][obstacle_class]}
障碍物置信度: {obstacle_conf:.4f}
风险评估等级: {risk_level}
风险评估置信度: {risk_conf:.4f}
建议行动: {decision}
行动置信度: {decision_conf:.4f}

详细概率分布:
障碍物概率分布: {results['obstacle'].get('probabilities', 'N/A') if results['obstacle'] is not None else 'N/A'}
运动预测概率分布: {results['motion'].get('probabilities', 'N/A') if results['motion'] is not None else 'N/A'}
风险评估概率分布: {results['risk'].get('probabilities', 'N/A') if results['risk'] is not None else 'N/A'}
"""
        self.result_display.config(state=tk.NORMAL)
        self.result_display.delete(1.0, tk.END)
        self.result_display.insert(tk.END, result_text)
        self.result_display.config(state=tk.DISABLED)

    def convert_cv_to_photo(self, cv_img):
        """将OpenCV图像转换为PhotoImage，并等比缩放至填满显示区域"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # 获取图像和容器尺寸
        img_width, img_height = pil_image.size
        container_width, container_height = 600, 300  # 固定容器大小
        
        # 计算缩放比例，保持纵横比
        scale_width = container_width / img_width
        scale_height = container_height / img_height
        scale = max(scale_width, scale_height)  # 取较大的比例以填满容器
        
        # 计算新尺寸
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 等比缩放图像
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return ImageTk.PhotoImage(pil_image)

    def toggle_camera(self):
        """切换摄像头开关"""
        if self.camera_btn.cget("text") == "开启摄像头":
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """启动摄像头"""
        if self.camera_thread and self.camera_thread.is_alive():
            return

        self.camera_thread = CameraThread(self)
        self.camera_thread.start()

        self.camera_btn.config(text="关闭摄像头")
        self.log_message("摄像头已启动")
        self.status_bar.config(text="摄像头运行中")

    def stop_camera(self):
        """停止摄像头"""
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.stop()

        self.camera_btn.config(text="开启摄像头")
        self.camera_label.config(image="", text="摄像头画面将显示在这里")
        self.log_message("摄像头已停止")
        self.status_bar.config(text="就绪")

    def display_camera_frame(self, frame):
        """显示摄像头帧并进行实时处理"""
        if self.system:
            try:
                # 处理当前帧
                results = self.system.process_single_frame(frame)
                
                # 检查结果是否有效
                if results is None:
                    self.log_message("处理摄像头帧时返回空结果", level="warning")
                    photo = self.convert_cv_to_photo(frame)
                else:
                    # 在图像上绘制检测结果
                    display_frame = frame.copy()

                    # 绘制障碍物检测结果
                    obstacle_class = results['obstacle']['class'] if results['obstacle'] is not None else 0
                    obstacle_conf = results['obstacle']['confidence'] if results['obstacle'] is not None else 0.0
                    cv2.putText(display_frame,
                                f'Obstacle: {["无障碍物", "行人", "车辆"][obstacle_class]} ({obstacle_conf:.2f})',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 绘制风险评估结果
                    risk_level = results['risk']['level'] if results['risk'] is not None else "未知"
                    risk_conf = results['risk']['confidence'] if results['risk'] is not None else 0.0
                    cv2.putText(display_frame, f'Risk: {risk_level} ({risk_conf:.2f})',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 绘制避让决策结果
                    decision = results['decision']['action'] if results['decision'] is not None else "未知"
                    decision_conf = results['decision']['confidence'] if results['decision'] is not None else 0.0
                    cv2.putText(display_frame, f'Decision: {decision} ({decision_conf:.2f})',
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 更新结果显示区域
                    self.update_result_display(results)

                    # 转换为PhotoImage并显示
                    photo = self.convert_cv_to_photo(display_frame)
            except Exception as e:
                self.log_message(f"实时处理出错: {str(e)}", level="error")
                photo = self.convert_cv_to_photo(frame)
        else:
            photo = self.convert_cv_to_photo(frame)

        # 保存引用以防止被垃圾回收
        self.camera_image_photo = photo
        self.camera_label.config(image=photo, text="")

    def on_camera_error(self, error_message):
        """摄像头错误处理"""
        self.log_message(f"摄像头错误: {error_message}", level="error")
        self.stop_camera()
        messagebox.showerror("摄像头错误", error_message)

    def toggle_realtime_depth(self):
        """切换实时深度估计"""
        if self.depth_btn.cget("text") == "实时深度估计":
            self.start_realtime_depth()
        else:
            self.stop_realtime_depth()

    def start_realtime_depth(self):
        """启动实时深度估计"""
        if self.depth_thread and self.depth_thread.is_alive():
            return

        self.depth_thread = RealTimeDepthThread(self)
        self.depth_thread.start()

        self.depth_btn.config(text="停止深度估计")
        self.log_message("实时深度估计已启动")
        self.status_bar.config(text="实时深度估计运行中")

    def stop_realtime_depth(self):
        """停止实时深度估计"""
        if self.depth_thread and self.depth_thread.is_alive():
            self.depth_thread.stop()

        self.depth_btn.config(text="实时深度估计")
        self.depth_realtime_label.config(image="", text="实时深度估计将显示在这里")
        self.log_message("实时深度估计已停止")
        self.status_bar.config(text="就绪")

    def display_depth_frame(self, frame, risk_label, risk_confidence):
        """显示深度帧并更新风险评估信息"""
        if frame is None:
            self.log_message("无法显示空的深度帧", level="error")
            return
            
        photo = self.convert_cv_to_photo(frame)
        self.depth_realtime_image_photo = photo  # 保存引用
        self.depth_realtime_label.config(image=photo, text="")

        # 更新风险评估信息显示
        self.result_display.config(state=tk.NORMAL)
        current_text = self.result_display.get(1.0, tk.END)
        if current_text.strip():
            result_text = current_text.strip() + f"\n实时风险评估:\n风险等级: {risk_label}\n置信度: {risk_confidence:.4f}"
        else:
            result_text = f"实时风险评估:\n风险等级: {risk_label}\n置信度: {risk_confidence:.4f}"

        self.result_display.delete(1.0, tk.END)
        self.result_display.insert(tk.END, result_text)
        self.result_display.config(state=tk.DISABLED)

    def on_depth_error(self, error_message):
        """深度估计错误处理"""
        self.log_message(f"深度估计错误: {error_message}", level="error")
        self.stop_realtime_depth()
        messagebox.showerror("深度估计错误", error_message)

    def update_result_display(self, results):
        """更新结果显示区域"""
        obstacle_class = results['obstacle']['class']
        obstacle_conf = results['obstacle']['confidence']
        risk_level = results['risk']['level']
        risk_conf = results['risk']['confidence']
        decision = results['decision']['action']
        decision_conf = results['decision']['confidence']

        result_text = f"""处理结果摘要:
障碍物类型: {['无障碍物', '行人', '车辆'][obstacle_class]}
障碍物置信度: {obstacle_conf:.4f}
风险评估等级: {risk_level}
风险评估置信度: {risk_conf:.4f}
建议行动: {decision}
行动置信度: {decision_conf:.4f}

详细概率分布:
障碍物概率分布: {results['obstacle']['probabilities']}
运动预测概率分布: {results['motion']['probabilities']}
风险评估概率分布: {results['risk']['probabilities']}
"""
        self.result_display.config(state=tk.NORMAL)
        self.result_display.delete(1.0, tk.END)
        self.result_display.insert(tk.END, result_text)
        self.result_display.config(state=tk.DISABLED)

    def save_results(self):
        """保存结果"""
        self.result_display.config(state=tk.NORMAL)
        result_text = self.result_display.get(1.0, tk.END)
        self.result_display.config(state=tk.DISABLED)

        if not result_text.strip():
            messagebox.showwarning("警告", "没有结果可保存")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result_text)
                self.log_message(f"结果已保存到: {file_path}")
                messagebox.showinfo("成功", "结果保存成功")
            except Exception as e:
                error_msg = f"保存结果失败: {str(e)}"
                self.log_message(error_msg, level="error")
                messagebox.showerror("错误", error_msg)

    def clear_log(self):
        """清空日志"""
        self.log_display.config(state=tk.NORMAL)
        self.log_display.delete(1.0, tk.END)
        self.log_display.config(state=tk.DISABLED)
        self.log_message("日志已清空")
        
    def load_image_and_switch_tab(self):
        """加载图像并切换到原始图像标签页"""
        self.load_image()
        self.notebook.select(0)
        
    def process_image_and_switch_tab(self):
        """处理图像并切换到处理结果标签页"""
        self.process_image()
        self.notebook.select(1)
        
    def train_obstacle_and_switch_tab(self):
        """训练障碍物检测模型"""
        self.train_model("obstacle")
        
    def train_motion_and_switch_tab(self):
        """训练运动预测模型"""
        self.train_model("motion")
        
    def train_depth_and_switch_tab(self):
        """训练深度风险模型"""
        self.train_model("depth")
        
    def train_evasion_and_switch_tab(self):
        """训练避让决策模型"""
        self.train_model("evasion")
        
    def retrain_all_models_and_switch_tab(self):
        """全部重新训练"""
        self.retrain_all_models()
        
    def toggle_camera_and_switch_tab(self):
        """切换摄像头并切换到摄像头画面标签页"""
        self.toggle_camera()
        self.notebook.select(3)
        
    def toggle_realtime_depth_and_switch_tab(self):
        """切换实时深度估计并切换到实时深度估计标签页"""
        self.toggle_realtime_depth()
        self.notebook.select(4)

    def show_about(self):
        """显示关于对话框"""
        about_window = tk.Toplevel(self.root)
        about_window.title("关于")
        about_window.geometry("800x750")
        about_window.resizable(False, False)
        about_window.transient(self.root)
        about_window.grab_set()
        about_window.iconbitmap(os.path.join(get_base_path(), "img", "icon.ico"))

        # 设置窗口背景色
        about_window.configure(bg='#f8f9fa')

        # 创建标题区域
        header_frame = tk.Frame(about_window, bg='#343a40', height=150)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        # 主标题样式
        title_label = tk.Label(
            header_frame,
            text="叮丁多模态深度视觉避障系统",
            font=('微软雅黑', 20, 'bold'),
            fg='#ffffff',
            bg='#343a40'
        )
        title_label.pack(pady=(20, 5))

        # 副标题样式
        subtitle_label = tk.Label(
            header_frame,
            text="Depth Camera Obstacle Avoidance System",
            font=('微软雅黑', 15),
            fg='#ced4da',
            bg='#343a40'
        )
        subtitle_label.pack()

        # 版本信息
        version_label = tk.Label(
            header_frame,
            text="v1.0",
            font=('微软雅黑', 15, 'bold'),
            fg='#ffffff',
            bg='#343a40'
        )
        version_label.pack(pady=(5, 0))

        # 内容区域
        content_frame = tk.Frame(about_window, bg='#f8f9fa')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 创建卡片样式框架
        card_frame = tk.Frame(content_frame, bg='white', relief=tk.FLAT, bd=0)
        card_frame.pack(fill=tk.BOTH, expand=True)

        # 添加阴影效果
        card_border = tk.Frame(card_frame, bg='#dee2e6')
        card_border.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        inner_card = tk.Frame(card_border, bg='white')
        inner_card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # 内容文本区域
        text_frame = tk.Frame(inner_card, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 使用Text组件创建富文本效果
        text_area = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('微软雅黑', 10),
            bg='white',
            fg='#212529',
            bd=0,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )

        # 创建滚动条
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_area.yview)
        text_area.configure(yscrollcommand=scrollbar.set)

        # 布局
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 标题样式
        text_area.insert(tk.END, "系统介绍\n", "h2")
        text_area.insert(tk.END, "=" * 50 + "\n\n", "separator")
        intro_text = ("叮丁多模态深度视觉避障系统是一款基于计算机视觉和深度学习技术的智能避障解决方案。该系统通过深度相机捕获环境信息，"
                      "结合先进的机器学习算法，实现对障碍物的实时检测、运动轨迹预测、风险评估和避让决策。系统可广泛应用于"
                      "服务机器人、自动驾驶车辆、智能监控等领域。\n\n")
        text_area.insert(tk.END, intro_text, "p")

        # 功能列表样式
        text_area.insert(tk.END, "主要功能\n", "h2")
        text_area.insert(tk.END, "=" * 50 + "\n\n", "separator")
        features = [
            "障碍物检测：基于CNN-Swin Transformer混合模型，精准识别环境中的行人、车辆等障碍物",
            "运动预测：采用LSTM-Transformer混合模型，预测动态物体的运动轨迹",
            "深度估计：利用MiDaS模型进行深度估计，结合RGB图像进行风险评估",
            "避让决策：综合多维度信息，智能制定避让策略",
            "实时处理：支持实时摄像头输入，提供低延迟的避让决策",
            "可视化界面：提供直观的图形用户界面，便于操作和监控"
        ]

        for i, feature in enumerate(features, 1):
            text_area.insert(tk.END, f"{i}. {feature}\n", "li")
        text_area.insert(tk.END, "\n")

        # 技术栈样式
        text_area.insert(tk.END, "技术栈\n", "h2")
        text_area.insert(tk.END, "=" * 50 + "\n\n", "separator")
        tech_stack = [
            "核心框架：PyTorch深度学习框架",
            "界面技术：Tkinter GUI框架",
            "图像处理：OpenCV计算机视觉库",
            "开发语言：Python 3.9.6"
        ]

        for tech in tech_stack:
            text_area.insert(tk.END, f"• {tech}\n", "li")

        text_area.insert(tk.END, "\n模型架构：\n", "subheader")
        models = [
            "障碍物检测：CNN-Swin Transformer混合模型",
            "运动预测：LSTM-Transformer混合模型",
            "深度估计：MiDaS模型",
            "避让决策：多特征融合神经网络"
        ]

        for model in models:
            text_area.insert(tk.END, f"  - {model}\n", "li")
        text_area.insert(tk.END, "\n")

        # 开发信息样式
        text_area.insert(tk.END, "开发信息\n", "h2")
        text_area.insert(tk.END, "=" * 50 + "\n\n", "separator")
        dev_info = [
            "开发人员：吴迅",
            "开发团队：浙工大叮丁车队",
            "版权所有 © 2025 浙工大叮丁车队"
        ]

        for info in dev_info:
            text_area.insert(tk.END, f"{info}\n", "p")

        # 配置文本标签样式
        text_area.tag_configure("h2", font=('微软雅黑', 14, 'bold'), foreground='#343a40', spacing3=10)
        text_area.tag_configure("separator", font=('微软雅黑', 8), foreground='#ced4da')
        text_area.tag_configure("p", font=('微软雅黑', 10), spacing3=5, lmargin1=10, lmargin2=10)
        text_area.tag_configure("li", font=('微软雅黑', 10), lmargin1=20, lmargin2=20, spacing3=3)
        text_area.tag_configure("subheader", font=('微软雅黑', 10, 'bold'), lmargin1=20, lmargin2=20, spacing3=5)

        text_area.config(state=tk.DISABLED)

        # 底部按钮区域
        footer_frame = tk.Frame(about_window, bg='#f8f9fa')
        footer_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        # 关闭按钮样式
        close_btn = ModernButton(
            footer_frame,
            text="关闭",
            font=('微软雅黑', 10),
            width=12,
            height=1,
            bg_color='#007bff',
            hover_color='#0056b3',
            active_color='#004085',
            command=about_window.destroy
        )
        close_btn.pack(side=tk.RIGHT)


# ==================== 主函数 ====================
if __name__ == "__main__":
    # 设置日志
    logger = setup_logging()

    # 创建应用程序
    root = tk.Tk()
    root.title("叮丁多模态深度视觉避障系统")

    # 创建并显示主窗口
    app = ObstacleAvoidanceGUI(root)

    # 运行应用程序
    root.mainloop()
