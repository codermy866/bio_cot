import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    """温度缩放模块，用于置信度校准"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature

    def set_temperature(self, val):
        self.temperature.data = torch.tensor([val], dtype=self.temperature.dtype, device=self.temperature.device)

@torch.no_grad()
def tune_temperature(logits, labels, max_iter=50, lr=0.01, verbose=True):
    """
    自动拟合最优温度参数，使得softmax输出概率更可靠。
    logits: [N, num_classes]
    labels: [N]
    返回: 最优温度(float)
    """
    device = logits.device
    scaler = TemperatureScaler().to(device)
    nll_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    optimal_temp = scaler.temperature.item()
    if verbose:
        print(f"[Temperature Scaling] Optimal temperature: {optimal_temp:.4f}")
    return optimal_temp

def save_temperature(temp, path="optimal_temperature.txt"):
    with open(path, "w") as f:
        f.write(str(temp))

def load_temperature(path="optimal_temperature.txt"):
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 1.0  # 默认温度 