# core/obstacle_detection.py
import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchsummary import summary

# 项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 训练中断标志
training_interrupted = False


def window_partition(x, window_size):
    """
    将特征图划分为窗口
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    if H % window_size != 0 or W % window_size != 0:
        new_H = (H // window_size) * window_size
        new_W = (W // window_size) * window_size
        x = x[:, :new_H, :new_W, :]
        H, W = new_H, new_W

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口合并为特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始特征图高度
        W (int): 原始特征图宽度
    Returns:
        x: (B, H, W, C)
    """
    H = (H // window_size) * window_size
    W = (W // window_size) * window_size

    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """窗口多头自注意力"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # 相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            # 确保H和W是window_size的倍数
            H = (H // self.window_size) * self.window_size
            W = (W // self.window_size) * self.window_size

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        # 确保H和W是有效的，且H*W等于L
        if H * W != L:
            # 重新计算H和W
            H = W = int(L ** 0.5)
            # 确保H*W == L
            if H * W != L:
                H = int(L ** 0.5)
                W = L // H
                while H * W != L:
                    H -= 1
                    W = L // H

        assert H * W == L, f"input feature has wrong size: H={H}, W={W}, L={L}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch合并操作，用于下采样"""
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # 确保H和W是有效的
        if H * W != L:
            H = W = int(L ** 0.5)
            if H * W != L:
                H = int(L ** 0.5)
                W = L // H
                while H * W != L:
                    H -= 1
                    W = L // H

        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # 确保H和W是偶数
        if H % 2 != 0 or W % 2 != 0:
            # 裁剪到偶数尺寸
            H = (H // 2) * 2
            W = (W // 2) * 2
            x = x[:, :H, :W, :]

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """Swin Transformer基本层"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 构建Swin Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

        # Patch合并层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class AttentionBlock(nn.Module):
    """注意力机制块"""

    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)


class ObstacleDetectionCNN_SwinTransformer(nn.Module):
    """CNN-Swin Transformer混合模型"""
    def __init__(self, num_classes=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super(ObstacleDetectionCNN_SwinTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # 增强的CNN特征提取部分
        self.cnn_features = nn.Sequential(
            # 第一个卷积块 (7x7卷积，步长2)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 输出: 56x56
            AttentionBlock(64),

            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 28x28
            AttentionBlock(128),

            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 14x14
            AttentionBlock(256),

            # 第四个卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 不再下采样，保持14x14
            AttentionBlock(512),
        )

        # Patch embedding (1x1卷积保持尺寸)
        self.patch_embed = nn.Conv2d(512, embed_dim, kernel_size=1, stride=1)  # 输出: 14x14
        self.patch_resolution = (14, 14)

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建Swin Transformer层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = PatchMerging if (i_layer < self.num_layers - 1) else None

            # 计算每层的输入分辨率
            layer_input_resolution = (
                self.patch_resolution[0] // (2 ** i_layer),
                self.patch_resolution[1] // (2 ** i_layer)
            )

            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=layer_input_resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=min(window_size, min(layer_input_resolution)),
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=downsample)
            self.layers.append(layer)

        # Layer norm
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.classifier_dropout = nn.Dropout(0.5)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)),
                              num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """提取特征"""
        # CNN特征提取
        x = self.cnn_features(x)  # [B, 512, 14, 14]

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, 14, 14]

        # 转换为Transformer格式 [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, embed_dim]

        # Dropout
        x = self.pos_drop(x)

        # Swin Transformer阶段
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """前向传播"""
        x = self.forward_features(x)
        x = self.classifier_dropout(x)
        x = self.head(x)
        return x


class KITTIDetectionDataset(Dataset):
    """用于KITTI数据集的障碍物检测数据集"""
    def __init__(self, root_dir, image_set='training', transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, self.image_set, 'image_2')
        self.label_dir = os.path.join(self.root_dir, self.image_set, 'label_2')

        # 获取所有图像文件名
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.image_files.sort()

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取图像和对应的标签"""
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        # 加载标签
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.png', '.txt'))
        labels = self._parse_label(label_path)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 返回图像和标签
        return image, labels

    def _parse_label(self, label_path):
        """解析KITTI标签文件，返回类别标签"""
        if not os.path.exists(label_path):
            return 0  # 默认类别

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

        # 根据检测到的对象数量返回类别标签
        # 0: 无障碍物, 1: 行人, 2: 车辆
        if pedestrian_count > 0:
            return 1  # 行人
        elif car_count > 0:
            return 2  # 车辆
        else:
            return 0  # 无障碍物


def set_training_progress_callback(callback):
    """设置训练进度回调函数"""
    global training_progress_callback
    training_progress_callback = callback


def validate(model, val_loader, criterion, device):
    """验证函数"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return val_loss / len(val_loader), 100 * correct / total


def train_obstacle_detection(progress_callback=None):
    """训练障碍物检测模型"""
    global training_interrupted
    training_interrupted = False  # 重置中断标志

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 增强的数据预处理和数据增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset_path = os.path.join(project_root, 'dataset', 'kitti_object')
    if not os.path.exists(dataset_path):
        print("请先解压KITTI数据集到指定目录")
        return

    # 创建训练集和验证集
    full_dataset = KITTIDetectionDataset(root_dir=dataset_path, image_set='training', transform=train_transform)

    # 如果数据集为空，创建一个小的测试数据集
    if len(full_dataset) == 0:
        print("警告: 数据集为空，使用模拟数据进行测试")

        class DummyDataset(Dataset):
            def __init__(self, size=100, transform=None):
                self.size = size
                self.transform = transform or transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # 创建随机图像
                image = transforms.ToPILImage()(torch.rand(3, 224, 224))
                if self.transform:
                    image = self.transform(image)
                # 随机标签
                label = torch.randint(0, 3, (1,)).item()
                return image, label

        train_dataset = DummyDataset(size=80, transform=train_transform)
        val_dataset = DummyDataset(size=20, transform=val_transform)
    else:
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 创建模型
    model = ObstacleDetectionCNN_SwinTransformer(num_classes=3)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用标签平滑
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # 使用AdamW优化器

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 打印模型结构
    try:
        summary(model, (3, 224, 224))
    except Exception as e:
        print(f"模型摘要失败: {e}")
        print("继续训练...")

    # 训练模型
    num_epochs = 100
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    total_steps = num_epochs * len(train_loader)
    
    for epoch in range(num_epochs):
        # 检查中断标志
        if training_interrupted:
            print("训练已被用户中断")
            break

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # 检查中断标志
            if training_interrupted:
                print("训练已被用户中断")
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                progress_callback(progress, f"障碍物检测训练中 - Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}")

            # 打印训练进度
            accuracy = 100 * correct / total
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}, Accuracy: {accuracy:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

        # 如果被中断，跳出epoch循环
        if training_interrupted:
            break

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(project_root, 'models', 'best_obstacle_detection_model.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f'新最佳模型已保存，验证准确率: {best_val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停机制
        if patience_counter >= patience:
            print(f'验证准确率在{patience}个epoch内没有提升，提前停止训练')
            break

        # 更新学习率
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Learning rate: {scheduler.get_last_lr()[0]:.6f}')

    if training_interrupted:
        print("训练已被用户中断")
    else:
        print('障碍物检测模型训练完成')

        # 保存最终模型到models目录
        model_path = os.path.join(project_root, 'models', 'obstacle_detection_model.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f'最终模型已保存到: {model_path}')
        print(f'最佳验证准确率: {best_val_acc:.2f}%')


"""if __name__ == '__main__':
    # 调用训练函数
    train_obstacle_detection()"""