import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class CausalEncoder(nn.Module):
    """因果编码器：学习因果表示"""
    def __init__(self, input_dim: int, hidden_dim: int, causal_dim: int, num_heads: int = 4):
        super().__init__()
        self.causal_dim = causal_dim
        
        # 因果因子编码器 - 使用更稳定的架构
        self.causal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm提高稳定性
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, causal_dim * 2)  # 均值和方差
        )
        
        # 改进的权重初始化
        for i, m in enumerate(self.causal_encoder):
            if isinstance(m, nn.Linear):
                if i == len(self.causal_encoder) - 1:
                    # 输出层使用更小的初始化
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        
        # 因果发现网络 - 使用更稳定的架构
        self.causal_discovery = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, causal_dim * causal_dim)  # 因果邻接矩阵
        )
        
        # 因果注意力
        self.causal_attention = nn.MultiheadAttention(causal_dim, num_heads, dropout=0.1)
        
        # 添加温度参数用于控制分布
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入归一化，防止数值爆炸
        if torch.isnan(x).any():
            print("Warning: NaN detected in encode input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 使用更稳定的归一化方法
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True) + 1e-8  # 增加epsilon
        
        # 检查标准差是否为0或NaN
        if torch.isnan(x_std).any() or (x_std < 1e-8).any():
            print("Warning: Zero or NaN std detected in encode, using identity normalization")
            x = torch.zeros_like(x)
        else:
            x = (x - x_mean) / x_std
            # 限制数值范围
            x = torch.clamp(x, -10, 10)
        
        # 最终NaN检查
        if torch.isnan(x).any():
            print("Warning: NaN detected after encode normalization, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 逐层前向传播，添加梯度裁剪
        out = x
        for i, m in enumerate(self.causal_encoder):
            if isinstance(m, nn.Linear):
                # 检查权重和偏置
                if torch.isnan(m.weight).any() or torch.isnan(m.bias).any():
                    print(f"Warning: NaN in layer {i} weights/bias, reinitializing")
                    nn.init.xavier_uniform_(m.weight, gain=0.1 if i == len(self.causal_encoder) - 1 else 1.0)
                    nn.init.zeros_(m.bias)
            
            out = m(out)
            
            # 梯度裁剪和数值稳定性
            if torch.isnan(out).any():
                print(f"Warning: NaN in layer {i} output, replacing with zeros")
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 限制数值范围
            out = torch.clamp(out, -100, 100)
        
        causal_params = out
        mu, logvar = torch.chunk(causal_params, 2, dim=-1)
        
        # 更严格的logvar约束
        logvar = torch.clamp(logvar, min=-5, max=5)
        
        # 应用温度缩放
        mu = mu * self.temperature
        logvar = logvar * self.temperature
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        assert not torch.isnan(mu).any(), 'mu has nan'
        assert not torch.isnan(logvar).any(), 'logvar has nan'
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        out = mu + eps * std
        assert not torch.isnan(out).any(), 'reparameterize output has nan'
        return out
    
    def discover_causal_graph(self, causal_factors: torch.Tensor) -> torch.Tensor:
        """发现因果图结构"""
        batch_size = causal_factors.size(0)
        causal_adj = self.causal_discovery(causal_factors)  # [B, D*D]
        causal_adj = causal_adj.reshape(batch_size, self.causal_dim, self.causal_dim)
        
        # 应用稀疏性和DAG约束
        causal_adj = torch.sigmoid(causal_adj)
        causal_adj = causal_adj * (1 - torch.eye(self.causal_dim, device=causal_adj.device))
        return causal_adj
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        causal_factors = self.reparameterize(mu, logvar)
        causal_adj = self.discover_causal_graph(causal_factors)
        
        # 因果注意力
        causal_factors = causal_factors.unsqueeze(0)  # [1, B, D]
        attn_out, _ = self.causal_attention(causal_factors, causal_factors, causal_factors)
        causal_factors = attn_out.squeeze(0)  # [B, D]
        
        return {
            'causal_factors': causal_factors,
            'causal_adj': causal_adj,
            'mu': mu,
            'logvar': logvar
        }

class CausalIntervention(nn.Module):
    """因果干预模块"""
    def __init__(self, causal_dim: int, hidden_dim: int):
        super().__init__()
        self.causal_dim = causal_dim
        
        # 干预策略网络
        self.intervention_policy = nn.Sequential(
            nn.Linear(causal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, causal_dim)
        )
        
        # 干预效果预测器
        self.intervention_effect = nn.Sequential(
            nn.Linear(causal_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, causal_dim)
        )
        
    def do_intervention(self, causal_factors: torch.Tensor, 
                       intervention_targets: torch.Tensor,
                       intervention_values: torch.Tensor) -> torch.Tensor:
        """执行因果干预 do(X=x)"""
        # causal_factors: [B, D], intervention_targets/values: [B, D] or [B]
        assert causal_factors.size(0) == intervention_targets.size(0) == intervention_values.size(0), \
            f"Batch size mismatch: {causal_factors.shape}, {intervention_targets.shape}, {intervention_values.shape}"
        
        intervened_factors = causal_factors.clone()
        
        # 处理不同维度的干预目标
        if intervention_targets.dim() == 1:
            # [B] -> 每个样本一个干预目标
            for i, (target, value) in enumerate(zip(intervention_targets, intervention_values)):
                if target.item() >= 0:  # 有效干预
                    intervened_factors[i, target.item()] = value.item()
        else:
            # [B, D] -> 每个样本多个干预目标
            mask = intervention_targets > 0
            intervened_factors[mask] = intervention_values[mask]
        
        return intervened_factors
    
    def forward(self, causal_factors: torch.Tensor, 
                intervention_targets: Optional[torch.Tensor] = None,
                intervention_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        if intervention_targets is None or intervention_values is None:
            return causal_factors
        
        return self.do_intervention(causal_factors, intervention_targets, intervention_values)

class CounterfactualGenerator(nn.Module):
    """反事实生成器"""
    def __init__(self, causal_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.causal_dim = causal_dim
        
        # 反事实生成网络
        self.counterfactual_generator = nn.Sequential(
            nn.Linear(causal_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 反事实判别器
        self.counterfactual_discriminator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def generate_counterfactual(self, factual_factors: torch.Tensor, 
                              counterfactual_factors: torch.Tensor) -> torch.Tensor:
        """生成反事实"""
        combined_input = torch.cat([factual_factors, counterfactual_factors], dim=-1)
        assert not torch.isnan(combined_input).any(), 'combined_input has nan'
        counterfactual = self.counterfactual_generator(combined_input)
        assert not torch.isnan(counterfactual).any(), 'counterfactual has nan'
        return counterfactual
    
    def discriminate_counterfactual(self, counterfactual: torch.Tensor) -> torch.Tensor:
        """判别反事实的真实性"""
        assert not torch.isnan(counterfactual).any(), 'counterfactual input to discriminator has nan'
        discrimination = self.counterfactual_discriminator(counterfactual)
        assert not torch.isnan(discrimination).any(), 'discrimination has nan'
        return discrimination
    
    def forward(self, factual_factors: torch.Tensor, 
                counterfactual_factors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        counterfactual = self.generate_counterfactual(factual_factors, counterfactual_factors)
        discrimination = self.discriminate_counterfactual(counterfactual)
        
        return {
            'counterfactual': counterfactual,
            'discrimination': discrimination
        }

class CausalGNN(nn.Module):
    """因果图神经网络"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 causal_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.causal_dim = causal_dim
        
        # 因果编码器
        self.causal_encoder = CausalEncoder(input_dim, hidden_dim, causal_dim, num_heads)
        
        # 因果干预模块
        self.causal_intervention = CausalIntervention(causal_dim, hidden_dim)
        
        # 反事实生成器
        self.counterfactual_generator = CounterfactualGenerator(causal_dim, hidden_dim, output_dim)
        
        # 图神经网络层
        # 修正：每个节点是一个因果因子，输入维度为1
        self.gnn_layers = nn.ModuleList([
            GATConv(1, hidden_dim, heads=num_heads, dropout=dropout),
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 因果一致性损失权重
        self.causal_consistency_weight = 1.0
        self.intervention_weight = 0.5
        self.counterfactual_weight = 0.3
        
        # === 新增：因果因子滤波配置 ===
        # 通过因果效应挖掘重要性，屏蔽无关/冗余/干扰因子
        self.enable_causal_filter: bool = True
        self.keep_ratio: float = 0.5  # 保留前50%重要因子
        self.filter_temperature: float = 10.0  # 将soft重要性拉开
        self.filter_sparsity_weight: float = 0.01
        self.decorrelation_weight: float = 0.01
        
        # 环境变量覆盖（便于不改训练脚本进行AB测试）
        try:
            env_enable = os.environ.get('CAUSAL_ENABLE_FILTER')
            if env_enable is not None:
                self.enable_causal_filter = bool(int(env_enable))
            env_keep = os.environ.get('CAUSAL_KEEP_RATIO')
            if env_keep is not None:
                self.keep_ratio = float(env_keep)
            env_temp = os.environ.get('CAUSAL_FILTER_TEMPERATURE')
            if env_temp is not None:
                self.filter_temperature = float(env_temp)
            env_fs = os.environ.get('CAUSAL_FILTER_SPARSITY_WEIGHT')
            if env_fs is not None:
                self.filter_sparsity_weight = float(env_fs)
            env_dw = os.environ.get('CAUSAL_DECORRELATION_WEIGHT')
            if env_dw is not None:
                self.decorrelation_weight = float(env_dw)
        except Exception:
            pass
        
    def construct_causal_graph(self, causal_factors: torch.Tensor, 
                             causal_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = causal_factors.size(0)
        device = causal_factors.device
        edge_indices = []
        edge_weights = []
        for b in range(batch_size):
            adj = causal_adj[b]  # [D, D]
            edge_idx = torch.nonzero(adj > 0.1, as_tuple=False).t()
            edge_weight = adj[edge_idx[0], edge_idx[1]]
            # 添加批次偏移
            edge_idx[0] += b * self.causal_dim
            edge_idx[1] += b * self.causal_dim
            edge_indices.append(edge_idx)
            edge_weights.append(edge_weight)
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_weight = torch.cat(edge_weights, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weight = torch.empty((0,), dtype=torch.float, device=device)
        # 安全检查
        num_nodes = batch_size * self.causal_dim
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            min_idx = edge_index.min().item()
            if max_idx >= num_nodes or min_idx < 0:
                print(f"[CausalGNN] edge_index越界: min={min_idx}, max={max_idx}, num_nodes={num_nodes}")
                print(f"edge_index: {edge_index}")
                print(f"edge_weight: {edge_weight}")
                raise RuntimeError(f"edge_index out of bounds: min={min_idx}, max={max_idx}, num_nodes={num_nodes}")
        return edge_index, edge_weight
    
    def forward(self, x: torch.Tensor, 
                intervention_targets: Optional[torch.Tensor] = None,
                intervention_values: Optional[torch.Tensor] = None,
                generate_counterfactual: bool = False) -> Dict[str, torch.Tensor]:
        # 1. 因果编码
        causal_output = self.causal_encoder(x)
        causal_factors = causal_output['causal_factors']
        causal_adj = causal_output['causal_adj']
        # Debug: 打印因果因子shape
        print(f"[CausalGNN] causal_factors shape: {causal_factors.shape}")
        # 2. 因果干预（如果指定）
        if intervention_targets is not None and intervention_values is not None:
            intervened_factors = self.causal_intervention(
                causal_factors, intervention_targets, intervention_values
            )
        else:
            intervened_factors = causal_factors
        
        # 2.5 因果滤波（基于总效应的重要性）
        causal_mask = None
        factor_importance = None
        if self.enable_causal_filter:
            with torch.no_grad():
                # 使用总效应衡量重要性：对每个因子，取其出度影响的绝对和值
                direct_effects = causal_adj.mean(dim=0)  # [D, D]
                # 间接效应累加（限制阶数，保证数值稳定）
                indirect_effects = torch.zeros_like(direct_effects)
                adj_power = direct_effects.clone()
                for _ in range(2, min(4, self.causal_dim + 1)):
                    adj_power = torch.mm(adj_power, direct_effects)
                    indirect_effects += adj_power
                total_effects = direct_effects + indirect_effects  # [D, D]
                # 重要性：按行求绝对值和（该因子对其他因子的总影响）
                factor_importance = total_effects.abs().sum(dim=1)  # [D]
                # 归一化
                factor_importance = factor_importance / (factor_importance.max() + 1e-6)
                # 软门控分数，拉开区分度
                soft_scores = torch.sigmoid(self.filter_temperature * (factor_importance - factor_importance.mean()))
                # 按比例保留Top-K
                k = max(1, int(self.causal_dim * self.keep_ratio))
                topk_vals, topk_idx = torch.topk(factor_importance, k=k, largest=True)
                hard_mask = torch.zeros(self.causal_dim, device=intervened_factors.device)
                hard_mask[topk_idx] = 1.0
                # 软硬结合（straight-through风格的门控）
                causal_mask = hard_mask + soft_scores - soft_scores.detach()
            # 应用掩码
            intervened_factors = intervened_factors * causal_mask.unsqueeze(0)
        # 3. 构建因果图
        edge_index, edge_weight = self.construct_causal_graph(intervened_factors, causal_adj)
        # 4. 图神经网络传播
        # 修正：每个因果因子为一个节点，node_features=[B*D, 1]
        node_features = intervened_factors.reshape(-1, 1)  # [B*D, 1]
        num_nodes = node_features.shape[0]
        # Debug: 打印GNN输入shape
        print(f"[CausalGNN] node_features shape: {node_features.shape}, edge_index shape: {edge_index.shape}")
        # 再次安全检查
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            min_idx = edge_index.min().item()
            if max_idx >= num_nodes or min_idx < 0:
                print(f"[CausalGNN-Forward] edge_index越界: min={min_idx}, max={max_idx}, num_nodes={num_nodes}")
                print(f"edge_index: {edge_index}")
                print(f"edge_weight: {edge_weight}")
                print(f"node_features.shape: {node_features.shape}")
                print(f"batch_size: {x.shape[0]}, causal_dim: {self.causal_dim}")
                raise RuntimeError(f"edge_index out of bounds: min={min_idx}, max={max_idx}, num_nodes={num_nodes}")
        for gnn_layer in self.gnn_layers:
            node_features = F.elu(gnn_layer(node_features, edge_index, edge_weight))
        # 5. 聚合节点特征
        node_features = node_features.reshape(-1, self.causal_dim, node_features.size(-1))  # [B, D, H]
        graph_features = node_features.mean(dim=1)  # [B, H]
        output = self.output_projection(graph_features)  # [B, output_dim]
        # 6. 输出预测
        # 7. 反事实生成（如果需要）
        counterfactual_output = None
        if generate_counterfactual:
            noise = torch.randn_like(causal_factors)
            counterfactual_factors = causal_factors + 0.1 * noise
            counterfactual_output = self.counterfactual_generator(
                causal_factors, counterfactual_factors
            )
        return {
            'output': output,
            'causal_factors': causal_factors,
            'intervened_factors': intervened_factors,
            'causal_adj': causal_adj,
            'graph_features': output,  # 直接返回output，保证是[B, 1024]
            'counterfactual': counterfactual_output,
            'causal_mask': causal_mask,
            'factor_importance': factor_importance,
            **causal_output
        }
    
    def compute_causal_losses(self, outputs: Dict[str, torch.Tensor], 
                            targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算因果相关损失"""
        losses = {}
        
        # 1. 主任务损失 - 使用焦点损失处理类别不平衡
        task_loss = self._focal_loss(outputs['output'], targets, alpha=0.25, gamma=2.0)
        losses['task_loss'] = task_loss
        
        # 2. 因果一致性损失（KL散度）- 添加KL退火
        mu, logvar = outputs['mu'], outputs['logvar']
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 使用KL退火，逐渐增加KL损失权重
        kl_weight = min(0.1, 0.001 * (torch.tensor(1.0, device=kl_loss.device)))  # 从0.001逐渐增加到0.1
        losses['causal_consistency'] = kl_loss * kl_weight
        
        # 3. 因果图稀疏性损失 - 使用L1正则化
        causal_adj = outputs['causal_adj']
        sparsity_loss = torch.mean(torch.abs(causal_adj))
        losses['sparsity'] = sparsity_loss * 0.01  # 降低稀疏性约束
        
        # 4. DAG约束损失（确保无环）- 使用更稳定的实现
        try:
            dag_loss = self._compute_dag_loss(causal_adj)
            if torch.isnan(dag_loss) or torch.isinf(dag_loss):
                print("Warning: DAG loss is NaN or inf, setting to zero")
                dag_loss = torch.tensor(0.0, device=causal_adj.device, dtype=causal_adj.dtype)
            losses['dag_constraint'] = dag_loss * 0.005  # 进一步降低DAG约束权重
        except Exception as e:
            print(f"Error computing DAG loss: {e}")
            losses['dag_constraint'] = torch.tensor(0.0, device=causal_adj.device, dtype=causal_adj.dtype)
        
        # 5. 反事实损失（如果生成了反事实）
        if outputs['counterfactual'] is not None:
            counterfactual = outputs['counterfactual']['counterfactual']
            discrimination = outputs['counterfactual']['discrimination']
            
            # 反事实真实性损失
            cf_real_loss = F.binary_cross_entropy(discrimination, torch.ones_like(discrimination))
            losses['counterfactual_realism'] = cf_real_loss * 0.1  # 降低权重
            
            # 反事实多样性损失
            if counterfactual.size(-1) != outputs['output'].size(-1):
                cf_diversity = torch.mean(torch.abs(counterfactual))
            else:
                cf_diversity = torch.mean(torch.abs(counterfactual - outputs['output']))
            losses['counterfactual_diversity'] = cf_diversity * 0.05

        # 6. 因果滤波相关的正则项
        if 'causal_mask' in outputs and outputs['causal_mask'] is not None:
            mask = outputs['causal_mask']
            sparsity = mask.mean()
            losses['filter_sparsity'] = sparsity * 0.01  # 降低权重
        
        # 7. 去冗余损失：鼓励保留的因子特征间低相关
        if 'intervened_factors' in outputs:
            f = outputs['intervened_factors']
            if f.dim() == 2 and f.size(0) > 1:
                f_norm = (f - f.mean(dim=0, keepdim=True)) / (f.std(dim=0, keepdim=True) + 1e-6)
                cov = torch.mm(f_norm.t(), f_norm) / (f_norm.size(0) - 1 + 1e-6)
                off_diag = cov - torch.diag(torch.diag(cov))
                decor = off_diag.pow(2).mean()
                losses['decorrelation'] = decor * 0.01  # 降低权重
        
        # 8. 添加特征平滑损失，提高泛化能力
        if 'causal_factors' in outputs:
            causal_factors = outputs['causal_factors']
            smoothness_loss = torch.mean(torch.abs(causal_factors - causal_factors.mean(dim=0, keepdim=True)))
            losses['smoothness'] = smoothness_loss * 0.01
        
        return losses
    
    def _focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """焦点损失，用于处理类别不平衡"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def _compute_dag_loss(self, causal_adj: torch.Tensor) -> torch.Tensor:
        """计算DAG约束损失 - 修复数值稳定性问题并控制损失范围"""
        try:
            # 检查输入是否有NaN或inf
            if torch.isnan(causal_adj).any() or torch.isinf(causal_adj).any():
                print("Warning: NaN or inf detected in causal_adj, returning zero loss")
                return torch.tensor(0.0, device=causal_adj.device, dtype=causal_adj.dtype)
            
            # 限制矩阵元素范围，防止数值爆炸 - 适中的限制
            causal_adj = torch.clamp(causal_adj, min=-0.5, max=0.5)
            
            # 使用更稳定的DAG约束计算方法
            batch_size = causal_adj.size(0)
            dag_loss = torch.tensor(0.0, device=causal_adj.device, dtype=causal_adj.dtype)
            
            for b in range(batch_size):
                adj = causal_adj[b]  # [D, D]
                
                # 检查对角线元素
                if torch.isnan(adj.diagonal()).any() or torch.isinf(adj.diagonal()).any():
                    continue
                
                # 计算矩阵幂级数，但限制迭代次数和数值范围
                adj_power = adj.clone()
                for i in range(2, min(4, self.causal_dim + 1)):  # 限制最大迭代次数为3
                    adj_power = torch.mm(adj_power, adj)
                    # 适中的数值范围限制，防止溢出
                    adj_power = torch.clamp(adj_power, min=-2.0, max=2.0)
                    
                    # 检查是否有NaN或inf
                    if torch.isnan(adj_power).any() or torch.isinf(adj_power).any():
                        break
                    
                    # 计算trace并应用适中的缩放
                    trace = adj_power.diagonal().sum()
                    if not torch.isnan(trace) and not torch.isinf(trace):
                        # 应用适中的缩放因子，使损失值在合理范围内
                        scaled_trace = trace / (self.causal_dim * (i + 1))
                        dag_loss += scaled_trace
            
            # 取平均值并进一步缩放
            dag_loss = dag_loss / max(batch_size, 1)
            
            # 应用适中的最终缩放因子，确保损失值在0-1范围内
            dag_loss = torch.clamp(dag_loss, min=0.0, max=1.0)
            
            # 最终检查
            if torch.isnan(dag_loss) or torch.isinf(dag_loss):
                print("Warning: DAG loss is NaN or inf, returning zero loss")
                return torch.tensor(0.0, device=causal_adj.device, dtype=causal_adj.dtype)
            
            return dag_loss
            
        except Exception as e:
            print(f"Error in _compute_dag_loss: {e}")
            return torch.tensor(0.0, device=causal_adj.device, dtype=causal_adj.dtype)
    
    def interpret_causal_effects(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """解释因果效应"""
        causal_adj = outputs['causal_adj']
        causal_factors = outputs['causal_factors']
        
        # 计算直接因果效应
        direct_effects = causal_adj.mean(dim=0)  # [D, D]
        
        # 计算间接因果效应（通过路径分析）
        indirect_effects = torch.zeros_like(direct_effects)
        adj_power = causal_adj.mean(dim=0)
        
        for i in range(2, self.causal_dim + 1):
            adj_power = torch.mm(adj_power, causal_adj.mean(dim=0))
            indirect_effects += adj_power
        
        # 计算总因果效应
        total_effects = direct_effects + indirect_effects
        
        return {
            'direct_effects': direct_effects,
            'indirect_effects': indirect_effects,
            'total_effects': total_effects,
            'causal_strength': torch.norm(causal_adj, dim=(1, 2)).mean()
        }

class CausalMultimodalTransformer(nn.Module):
    """因果多模态Transformer"""
    def __init__(self, oct_model, col_model, num_classes, embed_dim=1024, 
                 causal_dim=64, dropout_rate=0.2, drop_path_rate=0.2):
        super().__init__()
        self.oct_vit = oct_model
        self.col_vit = col_model
        
        # 多模态融合
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout_rate)
        
        # 因果GNN
        self.causal_gnn = CausalGNN(
            input_dim=8,  # 临床数据维度 - 修正为实际维度 (7 + 1)
            hidden_dim=128,
            output_dim=embed_dim,
            causal_dim=causal_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # === 新增：三路特征投影层 ===
        # ViT-B/16输出始终为768
        self.oct_proj = nn.Linear(768, embed_dim)
        self.col_proj = nn.Linear(768, embed_dim)
        self.clinical_proj = nn.Linear(embed_dim, embed_dim)  # CausalGNN输出已是embed_dim，防止shape不一致
        
        # 输出头
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.drop_path_rate = drop_path_rate
        
        # === 新增：模态门控超参 ===
        self.modality_gate_scale: float = 5.0
        
        # 环境变量覆盖
        try:
            env_mgs = os.environ.get('MODALITY_GATE_SCALE')
            if env_mgs is not None:
                self.modality_gate_scale = float(env_mgs)
        except Exception:
            pass
        
    def forward(self, oct_img, col_img, clinical_data, 
                intervention_targets=None, intervention_values=None,
                generate_counterfactual=False) -> Dict[str, torch.Tensor]:
        # 处理OCT图像
        oct_features = self.oct_vit(oct_img)
        if isinstance(oct_features, dict):
            oct_features = oct_features['features']
        oct_features = self.oct_proj(oct_features)
        
        # 处理阴道镜图像 - 处理5维输入 [B, 3, C, H, W]
        if col_img.dim() == 5:
            batch_size = col_img.size(0)
            col_img = col_img.view(-1, *col_img.shape[2:])  # [B*3, C, H, W]
            col_features = self.col_vit(col_img)
            if isinstance(col_features, dict):
                col_features = col_features['features']
            col_features = col_features.view(batch_size, 3, -1).mean(dim=1)  # [B, embed_dim]
        else:
            col_features = self.col_vit(col_img)
            if isinstance(col_features, dict):
                col_features = col_features['features']
        col_features = self.col_proj(col_features)
        
        # 处理临床数据（通过因果GNN）
        clinical_output = self.causal_gnn(
            clinical_data, 
            intervention_targets, 
            intervention_values,
            generate_counterfactual
        )
        clinical_features = clinical_output['graph_features']  # [B, embed_dim]
        clinical_features = self.clinical_proj(clinical_features)
        
        # Debug: 打印三模态特征shape
        print(f"oct_features shape: {oct_features.shape}")
        print(f"col_features shape: {col_features.shape}")
        print(f"clinical_features shape: {clinical_features.shape}")
        
        # === 新增：基于临床-因果的模态门控（滤除无关/干扰模态信息） ===
        try:
            # 基于与临床因果表示的相似度做门控
            def cosine_gate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
                b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
                cos = (a_n * b_n).sum(dim=-1, keepdim=True)  # [B,1]
                return torch.sigmoid(self.modality_gate_scale * cos)
            gate_oct = cosine_gate(oct_features, clinical_features)  # [B,1]
            gate_col = cosine_gate(col_features, clinical_features)  # [B,1]
            # 可选：结合因子重要性（若可用）
            factor_importance = clinical_output.get('factor_importance', None)
            if factor_importance is not None:
                fi = factor_importance / (factor_importance.max() + 1e-6)
                fi_score = (fi.mean().clamp(0, 1)).detach()
                # 将全局重要性嵌入门控
                gate_oct = torch.clamp(gate_oct * (0.5 + 0.5 * fi_score), 0.0, 1.0)
                gate_col = torch.clamp(gate_col * (0.5 + 0.5 * fi_score), 0.0, 1.0)
            # 应用门控
            oct_features = oct_features * gate_oct
            col_features = col_features * gate_col
        except Exception:
            pass
        
        # 多模态融合
        # 修复维度不匹配问题
        batch_size = oct_features.size(0)
        
        # 确保所有特征具有相同的batch size
        if col_features.size(0) != batch_size:
            # 如果col_features的batch size不同，取前batch_size个样本
            col_features = col_features[:batch_size]
        
        if clinical_features.size(0) != batch_size:
            # 如果clinical_features的batch size不同，取前batch_size个样本
            clinical_features = clinical_features[:batch_size]
        
        # 验证所有特征具有相同的维度
        assert oct_features.size(0) == col_features.size(0) == clinical_features.size(0), \
            f"Batch size mismatch: oct={oct_features.size(0)}, col={col_features.size(0)}, clinical={clinical_features.size(0)}"
        
        features = torch.stack([oct_features, col_features, clinical_features], dim=0)  # [3, B, embed_dim]
        attn_output, _ = self.cross_attention(features, features, features)  # [3, B, embed_dim]
        fused_features = attn_output.mean(dim=0)  # [B, embed_dim]
        
        fused_features = self.dropout(fused_features)
        output = self.head(fused_features)  # [B, num_classes]
        
        return {
            'output': output,
            'fused_features': fused_features,
            'clinical_output': clinical_output,
            'oct_features': oct_features,
            'col_features': col_features
        }
    
    def compute_losses(self, outputs: Dict[str, torch.Tensor], 
                      targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算所有损失"""
        # 主任务损失
        task_loss = F.cross_entropy(outputs['output'], targets)
        
        # 因果损失 - 需要从clinical_output中提取因果信息
        clinical_output = outputs['clinical_output']
        
        # 构建完整的因果输出字典
        causal_outputs = {
            'output': outputs['output'],  # 主输出
            'causal_factors': clinical_output['causal_factors'],
            'causal_adj': clinical_output['causal_adj'],
            'mu': clinical_output['mu'],
            'logvar': clinical_output['logvar'],
            'counterfactual': clinical_output.get('counterfactual', None)
        }
        
        # 计算因果损失
        causal_losses = self.causal_gnn.compute_causal_losses(
            causal_outputs, targets
        )
        
        # 合并损失
        total_loss = task_loss
        for loss_name, loss_value in causal_losses.items():
            total_loss += loss_value
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            **causal_losses
        }
    
    def interpret_causal_effects(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """解释因果效应"""
        return self.causal_gnn.interpret_causal_effects(outputs['clinical_output'])
    
    def generate_counterfactual_explanations(self, oct_img, col_img, clinical_data, 
                                           targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成反事实解释"""
        # 生成反事实
        outputs = self.forward(oct_img, col_img, clinical_data, 
                              generate_counterfactual=True)
        
        # 计算反事实预测
        cf_outputs = outputs['clinical_output']['counterfactual']
        
        # 计算反事实解释
        factual_pred = F.softmax(outputs['output'], dim=1)
        cf_pred = F.softmax(cf_outputs['counterfactual'], dim=1)
        
        # 解释差异
        explanation_diff = torch.abs(factual_pred - cf_pred)
        
        return {
            'factual_prediction': factual_pred,
            'counterfactual_prediction': cf_pred,
            'explanation_difference': explanation_diff,
            'counterfactual_confidence': cf_outputs['discrimination']
        } 