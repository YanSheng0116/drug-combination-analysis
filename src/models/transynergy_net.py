# src/models/transsynergy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import cfg

class TranSynergyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 从配置中读取网络超参数
        params     = cfg.TRANSYNERGY_PARAMS
        input_dim  = params.get("input_dim", 9608)
        hidden_dim = params.get("hidden_dim", 2048)
        dropout    = params.get("dropout", 0.5)

        # 第一层： input_dim → hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dp1 = nn.Dropout(dropout)

        # 第二层： hidden_dim → hidden_dim//2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dp2 = nn.Dropout(dropout)

        # 第三层： hidden_dim//2 → hidden_dim//4
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dp3 = nn.Dropout(dropout)

        # 第四层： hidden_dim//4 → 1（回归输出）
        self.fc4 = nn.Linear(hidden_dim // 4, 1)

    def forward(self, x):
        # Layer 1
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)

        # Layer 2
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dp2(x)

        # Layer 3
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dp3(x)

        # 输出层
        out = self.fc4(x)
        return out