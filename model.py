# model.py
# -*- coding: utf-8 -*-
"""
生产环境模型
输入: (b, 1, n, d)
输出: (b, n)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

# 导入预处理器（head1-5使用preprocessor_nlogwmz1，head6-7使用preprocessor_nlogwmz3）
from preprocessor_nlogwmz1 import PreprocessorNLogWMZ as PreprocessorNLogWMZ1
from preprocessor_nlogwmz3 import PreprocessorNLogWMZ as PreprocessorNLogWMZ3


# ========= 模型结构 =========
class MultitaskBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [6020, 6020, 2048], dropout: float = 0.45, use_bn: bool = False, use_hidden_layernorm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.use_bn = use_bn
        self.use_hidden_layernorm = use_hidden_layernorm
        self.input_bn = nn.BatchNorm1d(input_dim) if use_bn else nn.Identity()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_hidden_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 4:
            b, seq, n, f = x.shape
            x = x.reshape(b * seq * n, f)
            need_reshape = True
            target_shape = (b, seq, n, self.output_dim)
        elif x.dim() == 3:
            seq, n, f = x.shape
            x = x.reshape(seq * n, f)
            need_reshape = True
            target_shape = (seq, n, self.output_dim)
        elif x.dim() == 2:
            n, f = x.shape
            need_reshape = True
            target_shape = (n, self.output_dim)
        else:
            need_reshape = False
        if self.use_bn and x.dim() == 2:
            x = self.input_bn(x)
        shared_repr = self.net(x)
        if need_reshape:
            shared_repr = shared_repr.reshape(target_shape)
        return shared_repr


class TaskHead(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256, dropout: float = 0.0, use_layernorm: bool = True):
        super().__init__()
        self.use_layernorm = use_layernorm
        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        original_shape = shared_repr.shape
        if shared_repr.dim() == 4:
            b, seq, n, d = shared_repr.shape
            shared_repr = shared_repr.reshape(b * seq * n, d)
            need_reshape = True
            target_shape = (b, seq, n)
        elif shared_repr.dim() == 3:
            seq, n, d = shared_repr.shape
            shared_repr = shared_repr.reshape(seq * n, d)
            need_reshape = True
            target_shape = (seq, n)
        elif shared_repr.dim() == 2:
            n, d = shared_repr.shape
            need_reshape = True
            target_shape = (n,)
        else:
            need_reshape = False
        pred = self.net(shared_repr)
        if need_reshape:
            pred = pred.squeeze(-1).reshape(target_shape)
        else:
            pred = pred.squeeze(-1)
        return pred


class MultitaskHeadModel(nn.Module):
    def __init__(self, backbone: MultitaskBackbone, head: TaskHead, preprocessor, preprocessor_config: dict, device: torch.device):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.preprocessor = preprocessor  # 预处理器实例
        self.preprocessor_config = preprocessor_config
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, n, d = x.shape
        valid_mask = torch.ones(b, seq, n, dtype=torch.bool, device=x.device)
        x_preprocessed = self.preprocessor.transform_batch(x, valid_mask, self.preprocessor_config, show_progress=False)
        shared_repr = self.backbone(x_preprocessed)
        pred = self.head(shared_repr)
        if pred.dim() == 3:
            pred = pred[:, -1, :]
        return pred


class MultitaskModel(nn.Module):
    def __init__(self, head1_model: MultitaskHeadModel, head2_model: MultitaskHeadModel, head3_model: MultitaskHeadModel, 
                 head4_model: MultitaskHeadModel, head5_model: MultitaskHeadModel,
                 head6_model: MultitaskHeadModel, head7_model: MultitaskHeadModel,
                 ensemble_weights: list = [0.05, 0.1, 0.2, 0.05, 0.2, 0.3, 0.1], device: torch.device = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.head1 = head1_model
        self.head2 = head2_model
        self.head3 = head3_model
        self.head4 = head4_model
        self.head5 = head5_model
        self.head6 = head6_model
        self.head7 = head7_model
        total_weight = sum(ensemble_weights)
        self.ensemble_weights = [w / total_weight for w in ensemble_weights]
        self.to(device)
        self.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred1 = self.head1(x)
        pred2 = self.head2(x)
        pred3 = self.head3(x)
        pred4 = self.head4(x)
        pred5 = self.head5(x)
        pred6 = self.head6(x)
        pred7 = self.head7(x)
        pred_final = (
            self.ensemble_weights[0] * pred1 +
            self.ensemble_weights[1] * pred2 +
            self.ensemble_weights[2] * pred3 +
            self.ensemble_weights[3] * pred4 +
            self.ensemble_weights[4] * pred5 +
            self.ensemble_weights[5] * pred6 +
            self.ensemble_weights[6] * pred7
        )
        return pred_final


# ========= 预处理配置（硬编码，从CONFIG-WILL-BE-DELETED中提取） =========
def get_preprocessor_config_head1():
    """head1: Box-Cox/YJ, clip_value=7.0"""
    return {
        "check_invalid_values": True,
        "exclude_filled_nan_in_stats": True,
        "set_filled_nan_to_zero": False,
        "enable_nan_fill": True,
        "enable_log": True,
        "enable_winsor": False,
        "enable_zscore_step1": True,
        "enable_clip": True,
        "winsor_low": 0.001,
        "winsor_high": 0.999,
        "clip_value": 7.0,
        "fusion": {
            "use_raw": False,
            "use_minmax": False,
            "use_zscore_step2": False,
            "use_rank": False,
            "use_boxcox_yj": True,
            "boxcox_yj": {
                "method": "yeo_johnson",
                "lambda": 0.9,
                "shift": 1.0e-6
            },
            "weight_raw": 0,
            "weight_minmax": 0,
            "weight_zscore": 0,
            "weight_rank": 0,
            "weight_boxcox_yj": 1
        }
    }


def get_preprocessor_config_head2():
    """head2: Min-Max, clip_value=18.0"""
    return {
        "check_invalid_values": True,
        "exclude_filled_nan_in_stats": True,
        "set_filled_nan_to_zero": False,
        "enable_nan_fill": True,
        "enable_log": True,
        "enable_winsor": False,
        "enable_zscore_step1": True,
        "enable_clip": True,
        "winsor_low": 0.001,
        "winsor_high": 0.999,
        "clip_value": 18.0,
        "fusion": {
            "use_raw": False,
            "use_minmax": True,
            "use_zscore_step2": False,
            "use_rank": False,
            "minmax": {
                "quantile_low": 0.009,
                "quantile_high": 0.992
            },
            "rank_method": "adaptive",
            "rank_adaptive": {
                "head_ratio": 0.35,
                "tail_ratio": 0.35,
                "head_buckets": 1800,
                "tail_buckets": 1800,
                "mid_buckets": 500
            },
            "weight_raw": 0,
            "weight_minmax": 1,
            "weight_zscore": 0,
            "weight_rank": 0
        }
    }


def get_preprocessor_config_head3():
    """head3: Min-Max, clip_value=18.0"""
    return {
        "check_invalid_values": True,
        "exclude_filled_nan_in_stats": True,
        "set_filled_nan_to_zero": False,
        "enable_nan_fill": True,
        "enable_log": True,
        "enable_winsor": False,
        "enable_zscore_step1": True,
        "enable_clip": True,
        "winsor_low": 0.001,
        "winsor_high": 0.999,
        "clip_value": 18.0,
        "fusion": {
            "use_raw": False,
            "use_minmax": True,
            "use_zscore_step2": False,
            "use_rank": False,
            "minmax": {
                "quantile_low": 0.009,
                "quantile_high": 0.992
            },
            "rank_method": "adaptive",
            "rank_adaptive": {
                "head_ratio": 0.35,
                "tail_ratio": 0.35,
                "head_buckets": 1800,
                "tail_buckets": 1800,
                "mid_buckets": 500
            },
            "weight_raw": 0,
            "weight_minmax": 1,
            "weight_zscore": 0,
            "weight_rank": 0
        }
    }


def get_preprocessor_config_head4():
    """head4: Rank (adaptive), clip_value=7.0"""
    return {
        "check_invalid_values": True,
        "exclude_filled_nan_in_stats": True,
        "set_filled_nan_to_zero": False,
        "enable_nan_fill": True,
        "enable_log": True,
        "enable_winsor": False,
        "enable_zscore_step1": True,
        "enable_clip": True,
        "winsor_low": 0.001,
        "winsor_high": 0.999,
        "clip_value": 7.0,
        "fusion": {
            "use_raw": False,
            "use_minmax": False,
            "use_zscore_step2": False,
            "use_rank": True,
            "rank_method": "adaptive",
            "rank_adaptive": {
                "head_ratio": 0.4,
                "tail_ratio": 0.4,
                "head_buckets": 2000,
                "tail_buckets": 2000,
                "mid_buckets": 300
            },
            "weight_raw": 0,
            "weight_minmax": 0,
            "weight_zscore": 0,
            "weight_rank": 1
        }
    }


def get_preprocessor_config_head5():
    """head5: Rank (uniform), clip_value=7.0"""
    return {
        "check_invalid_values": True,
        "exclude_filled_nan_in_stats": True,
        "set_filled_nan_to_zero": False,
        "enable_nan_fill": True,
        "enable_log": True,
        "enable_winsor": False,
        "enable_zscore_step1": True,
        "enable_clip": True,
        "winsor_low": 0.001,
        "winsor_high": 0.999,
        "clip_value": 7.0,
        "fusion": {
            "use_raw": False,
            "use_minmax": False,
            "use_zscore_step2": False,
            "use_rank": True,
            "rank_method": "uniform",
            "rank_adaptive": {
                "head_ratio": 0.4,
                "tail_ratio": 0.4,
                "head_buckets": 2000,
                "tail_buckets": 2000,
                "mid_buckets": 300
            },
            "weight_raw": 0,
            "weight_minmax": 0,
            "weight_zscore": 0,
            "weight_rank": 1
        }
    }


def get_preprocessor_config_head6():
    """head6: Min-Max with temporal norm (minmax), clip_value=18.0"""
    return {
        "check_invalid_values": True,
        "exclude_filled_nan_in_stats": True,
        "set_filled_nan_to_zero": False,
        "enable_nan_fill": True,
        "enable_log": True,
        "enable_winsor": False,
        "enable_zscore_step1": True,
        "enable_clip": True,
        "enable_temporal_norm": True,
        "temporal_norm": {
            "method": "minmax"
        },
        "winsor_low": 0.001,
        "winsor_high": 0.999,
        "clip_value": 18.0,
        "fusion": {
            "use_raw": False,
            "use_minmax": True,
            "use_zscore_step2": False,
            "use_rank": False,
            "minmax": {
                "quantile_low": 0.009,
                "quantile_high": 0.992
            },
            "rank_method": "adaptive",
            "rank_adaptive": {
                "head_ratio": 0.35,
                "tail_ratio": 0.35,
                "head_buckets": 1800,
                "tail_buckets": 1800,
                "mid_buckets": 500
            },
            "weight_raw": 0,
            "weight_minmax": 1,
            "weight_zscore": 0,
            "weight_rank": 0
        }
    }


def get_preprocessor_config_head7():
    """head7: Rank (adaptive) with temporal norm (zscore), clip_value=7.0"""
    return {
        "check_invalid_values": True,
        "exclude_filled_nan_in_stats": True,
        "set_filled_nan_to_zero": False,
        "enable_nan_fill": True,
        "enable_log": True,
        "enable_winsor": False,
        "enable_zscore_step1": True,
        "enable_clip": True,
        "enable_temporal_norm": True,
        "temporal_norm": {
            "method": "zscore"
        },
        "winsor_low": 0.001,
        "winsor_high": 0.999,
        "clip_value": 7.0,
        "fusion": {
            "use_raw": False,
            "use_minmax": False,
            "use_zscore_step2": False,
            "use_rank": True,
            "rank_method": "adaptive",
            "rank_adaptive": {
                "head_ratio": 0.4,
                "tail_ratio": 0.4,
                "head_buckets": 2000,
                "tail_buckets": 2000,
                "mid_buckets": 300
            },
            "weight_raw": 0,
            "weight_minmax": 0,
            "weight_zscore": 0,
            "weight_rank": 1
        }
    }


def load_model_checkpoint(model_path: str, model: nn.Module, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    return model


def load_multitask_model(backbone_path: str, head_path: str, preprocessor, preprocessor_config: dict, device: torch.device) -> MultitaskHeadModel:
    input_dim = 1771  # 新模型使用1771个因子
    backbone_hidden_dims = [6020, 6020, 2048]
    backbone_dropout = 0.45
    backbone_use_bn = False
    backbone_use_layernorm = True
    head_hidden_dim = 256
    head_dropout = 0.0
    head_use_layernorm = True
    
    backbone = MultitaskBackbone(
        input_dim=input_dim,
        hidden_dims=backbone_hidden_dims,
        dropout=backbone_dropout,
        use_bn=backbone_use_bn,
        use_hidden_layernorm=backbone_use_layernorm
    )
    
    head = TaskHead(
        input_dim=backbone_hidden_dims[-1],
        hidden_dim=head_hidden_dim,
        dropout=head_dropout,
        use_layernorm=head_use_layernorm
    )
    
    backbone = load_model_checkpoint(backbone_path, backbone, device)
    
    head_checkpoint = torch.load(head_path, map_location=device)
    if isinstance(head_checkpoint, dict):
        head_state_dict = {}
        task_idx = 0
        for key, value in head_checkpoint.items():
            if key.startswith(f"heads.{task_idx}."):
                new_key = key.replace(f"heads.{task_idx}.", "")
                head_state_dict[new_key] = value
            elif not key.startswith("backbone.") and not key.startswith("heads."):
                head_state_dict[key] = value
        head.load_state_dict(head_state_dict, strict=True)
    else:
        head.load_state_dict(head_checkpoint, strict=True)
    
    head.to(device)
    head.eval()
    
    multitask_head_model = MultitaskHeadModel(
        backbone=backbone,
        head=head,
        preprocessor=preprocessor,
        preprocessor_config=preprocessor_config,
        device=device
    )
    
    return multitask_head_model


def get_model() -> nn.Module:
    cur_folder = os.path.dirname(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 7个模型的路径
    head1_backbone_path = os.path.join(cur_folder, "MODEL-HEADS", "head1", "B.pth")
    head1_head_path = os.path.join(cur_folder, "MODEL-HEADS", "head1", "H.pth")
    head2_backbone_path = os.path.join(cur_folder, "MODEL-HEADS", "head2", "B.pth")
    head2_head_path = os.path.join(cur_folder, "MODEL-HEADS", "head2", "H.pth")
    head3_backbone_path = os.path.join(cur_folder, "MODEL-HEADS", "head3", "B.pth")
    head3_head_path = os.path.join(cur_folder, "MODEL-HEADS", "head3", "H.pth")
    head4_backbone_path = os.path.join(cur_folder, "MODEL-HEADS", "head4", "B.pth")
    head4_head_path = os.path.join(cur_folder, "MODEL-HEADS", "head4", "H.pth")
    head5_backbone_path = os.path.join(cur_folder, "MODEL-HEADS", "head5", "B.pth")
    head5_head_path = os.path.join(cur_folder, "MODEL-HEADS", "head5", "H.pth")
    head6_backbone_path = os.path.join(cur_folder, "MODEL-HEADS", "head6", "B.pth")
    head6_head_path = os.path.join(cur_folder, "MODEL-HEADS", "head6", "H.pth")
    head7_backbone_path = os.path.join(cur_folder, "MODEL-HEADS", "head7", "B.pth")
    head7_head_path = os.path.join(cur_folder, "MODEL-HEADS", "head7", "H.pth")
    
    # 获取预处理配置
    head1_preprocessor_config = get_preprocessor_config_head1()
    head2_preprocessor_config = get_preprocessor_config_head2()
    head3_preprocessor_config = get_preprocessor_config_head3()
    head4_preprocessor_config = get_preprocessor_config_head4()
    head5_preprocessor_config = get_preprocessor_config_head5()
    head6_preprocessor_config = get_preprocessor_config_head6()
    head7_preprocessor_config = get_preprocessor_config_head7()
    
    # 创建预处理器实例（head1-5使用preprocessor_nlogwmz1，head6-7使用preprocessor_nlogwmz3）
    preprocessor1 = PreprocessorNLogWMZ1()
    preprocessor3 = PreprocessorNLogWMZ3()
    
    # 加载7个模型
    head1_model = load_multitask_model(head1_backbone_path, head1_head_path, preprocessor1, head1_preprocessor_config, device)
    head2_model = load_multitask_model(head2_backbone_path, head2_head_path, preprocessor1, head2_preprocessor_config, device)
    head3_model = load_multitask_model(head3_backbone_path, head3_head_path, preprocessor1, head3_preprocessor_config, device)
    head4_model = load_multitask_model(head4_backbone_path, head4_head_path, preprocessor1, head4_preprocessor_config, device)
    head5_model = load_multitask_model(head5_backbone_path, head5_head_path, preprocessor1, head5_preprocessor_config, device)
    head6_model = load_multitask_model(head6_backbone_path, head6_head_path, preprocessor3, head6_preprocessor_config, device)
    head7_model = load_multitask_model(head7_backbone_path, head7_head_path, preprocessor3, head7_preprocessor_config, device)
    
    # 融合权重：0.05, 0.1, 0.2, 0.05, 0.2, 0.3, 0.1
    ensemble_weights = [0.05, 0.1, 0.2, 0.05, 0.2, 0.3, 0.1]
    
    model = MultitaskModel(
        head1_model=head1_model,
        head2_model=head2_model,
        head3_model=head3_model,
        head4_model=head4_model,
        head5_model=head5_model,
        head6_model=head6_model,
        head7_model=head7_model,
        ensemble_weights=ensemble_weights,
        device=device
    )
    
    return model


if __name__ == "__main__":
    model = get_model()
    batch_size = 2
    num_stocks = 100
    num_features = 1771  # 新模型使用1771个因子
    x = torch.randn(batch_size, 1, num_stocks, num_features).to(model.device)
    with torch.no_grad():
        pred = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {pred.shape}")
    print(f"预期输出形状: ({batch_size}, {num_stocks})")
    assert pred.shape == (batch_size, num_stocks), f"输出形状不匹配: {pred.shape} != ({batch_size}, {num_stocks})"
    print("测试通过！")