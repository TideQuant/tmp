# -*- coding: utf-8 -*-
"""
N-log-W-m-z 预处理模块（完全向量化版本 - 不考虑mask）
- 无全局参数，每个batch实时计算
- 处理顺序：异常值统一转NaN → NaN填充 → Log → 时序归一化(可选) → 百分比截断 → Z-score-step1 → 正负clip → 融合
- GPU加速，完全向量化处理（NaN填充后数据很齐，直接批量计算所有统计量）
- 支持均匀/非均匀分位数Rank
- 兼容旧版PyTorch（无nan*函数）
- Min-Max使用分位数，更鲁棒
- 扩展：robust_z / iqr_scale / boxcox|yeo_johnson / asinh|signed_log / tanh_estimator 通道可加权融合
- 新增：时序归一化功能，使用预计算的统计量对log后的切片进行归一化
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import os
import h5py


def _compute_adaptive_quantiles(head_ratio, tail_ratio, head_buckets, mid_buckets, tail_buckets, device):
    head_quantiles = torch.linspace(0.0, head_ratio, head_buckets + 1, device=device)
    mid_quantiles = torch.linspace(head_ratio, 1.0 - tail_ratio, mid_buckets + 1, device=device)
    tail_quantiles = torch.linspace(1.0 - tail_ratio, 1.0, tail_buckets + 1, device=device)
    return torch.cat([head_quantiles[:-1], mid_quantiles[:-1], tail_quantiles])


def _bucket_to_rank_adaptive(bucket_indices: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    max_bucket = len(quantiles) - 2
    bucket_indices = torch.clamp(bucket_indices, 0, max_bucket)
    bucket_lower = quantiles[bucket_indices]
    bucket_upper = quantiles[bucket_indices + 1]
    return (bucket_lower + bucket_upper) / 2.0


def _nanmedian_vectorized(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if hasattr(torch, 'nanmedian'):
        return torch.nanmedian(x, dim=dim)[0]
    x_sorted, _ = torch.sort(x, dim=dim)
    nan_count = torch.isnan(x).sum(dim=dim)
    valid_count = x.shape[dim] - nan_count
    median_idx = (valid_count - 1) // 2
    max_idx = x.shape[dim] - 1
    max_valid_idx = (valid_count - 1).long().clamp(min=0)
    median_idx = torch.clamp(median_idx, 0, torch.minimum(max_valid_idx, torch.tensor(max_idx, device=x.device, dtype=torch.long))).long()
    result = torch.gather(x_sorted, dim, median_idx.unsqueeze(dim))
    result = torch.where(valid_count > 0, result, torch.zeros_like(result))
    result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
    return result


def _nanmean_vectorized(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if hasattr(torch, 'nanmean'):
        return torch.nanmean(x, dim=dim)
    nan_mask = torch.isnan(x)
    x_clean = torch.where(nan_mask, torch.zeros_like(x), x)
    valid_count = (~nan_mask).sum(dim=dim).float()
    valid_count = torch.where(valid_count > 0, valid_count, torch.ones_like(valid_count))
    mean_vals = x_clean.sum(dim=dim) / valid_count
    mean_vals = torch.where(valid_count > 1, mean_vals, torch.zeros_like(mean_vals))
    return mean_vals


def _nanstd_vectorized(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if hasattr(torch, 'nanstd'):
        return torch.nanstd(x, dim=dim)
    nan_mask = torch.isnan(x)
    x_clean = torch.where(nan_mask, torch.zeros_like(x), x)
    valid_count = (~nan_mask).sum(dim=dim).float()
    valid_count = torch.where(valid_count > 0, valid_count, torch.ones_like(valid_count))
    mean_vals = x_clean.sum(dim=dim) / valid_count
    mean_vals = torch.where(valid_count > 1, mean_vals, torch.zeros_like(mean_vals))
    x_centered = x_clean - mean_vals.unsqueeze(dim)
    x_centered = torch.where(nan_mask, torch.zeros_like(x_centered), x_centered)
    variance = (x_centered ** 2).sum(dim=dim) / valid_count
    std_vals = torch.sqrt(variance)
    std_vals = torch.where(valid_count > 1, std_vals, torch.ones_like(std_vals))
    return std_vals


def _nanquantile_vectorized(x: torch.Tensor, q: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if hasattr(torch, 'nanquantile'):
        q = q.to(dtype=x.dtype)
        return torch.nanquantile(x, q, dim=dim)

    nan_mask = torch.isnan(x)
    x_sorted, _ = torch.sort(x, dim=dim)
    valid_count = (~nan_mask).sum(dim=dim).float()
    valid_count = torch.clamp(valid_count, min=1.0)
    if q.dim() == 0:
        q = q.unsqueeze(0)
    max_idx = x.shape[dim] - 1
    vals = []
    for qv in q:
        idx = (valid_count - 1) * qv
        idx_low = torch.floor(idx).long()
        idx_high = torch.ceil(idx).long()
        max_valid_idx = (valid_count - 1).long().clamp(min=0)
        max_idx_tensor = torch.minimum(max_valid_idx, torch.tensor(max_idx, device=x.device, dtype=torch.long))
        idx_low = torch.clamp(idx_low, 0, max_idx_tensor)
        idx_high = torch.clamp(idx_high, 0, max_idx_tensor)
        v_low = torch.gather(x_sorted, dim, idx_low.unsqueeze(dim))
        v_high = torch.gather(x_sorted, dim, idx_high.unsqueeze(dim))
        if v_low.dim() > 1:
            v_low = v_low.squeeze(dim)
        if v_high.dim() > 1:
            v_high = v_high.squeeze(dim)
        alpha = idx - idx_low.float()
        v = v_low * (1 - alpha) + v_high * alpha
        v = torch.where(valid_count > 0, v, torch.zeros_like(v))
        vals.append(v)
    return vals[0] if len(vals) == 1 else torch.stack(vals, dim=0)


class PreprocessorNLogWMZ:
    def __init__(self, stats_file_path: Optional[str] = None):
        """
        初始化预处理器
        
        参数:
            stats_file_path: 预计算统计量文件路径（可选）
        """
        self.stats_file_path = stats_file_path
        self.stats = None  # 预加载的统计量字典
        
        if stats_file_path is not None and os.path.exists(stats_file_path):
            self._load_stats(stats_file_path)
    
    def _load_stats(self, stats_file_path: str):
        """加载预计算的统计量文件"""
        print(f"【预处理】加载预计算统计量: {stats_file_path}")
        with h5py.File(stats_file_path, "r") as f:
            self.stats = {
                'mean': torch.from_numpy(f['mean'][:]).float(),      # (N, F)
                'std': torch.from_numpy(f['std'][:]).float(),        # (N, F)
                'min': torch.from_numpy(f['min'][:]).float(),        # (N, F)
                'max': torch.from_numpy(f['max'][:]).float(),        # (N, F)
                'median': torch.from_numpy(f['median'][:]).float(),  # (N, F)
            }
            print(f"【预处理】统计量加载完成:")
            for key, value in self.stats.items():
                print(f"  {key}: shape {value.shape}")
    
    def transform_batch(self, X: torch.Tensor, valid_mask: torch.Tensor, config: Dict, show_progress: bool = False) -> torch.Tensor:
        """
        对batch数据进行预处理（支持使用预计算统计量的时序归一化）
        """
        # 检查是否启用时序归一化（使用预计算统计量）
        enable_temporal_norm = config.get("enable_temporal_norm", False)
        use_precomputed_stats = enable_temporal_norm and self.stats is not None
        
        if X.dim() == 4:
            b, seq, N, F = X.shape
            Y_list = []
            total_slices = b * seq
            
            for b_idx in range(b):
                # 原有模式：逐个切片处理（现在支持预计算统计量的时序归一化）
                seq_list = []
                for s_idx in range(seq):
                    if show_progress:
                        idx = b_idx * seq + s_idx + 1
                        if idx == 88 or idx == total_slices:
                            print(f"\r【预处理进度】处理切片: {idx}/{total_slices} ({idx*1000//total_slices}%)", end="", flush=True)
                    
                    # 对每个切片应用预处理（如果启用时序归一化，会在log后使用预计算统计量）
                    Y_t = self.transform_slice_vectorized(X[b_idx, s_idx], config, use_precomputed_stats=use_precomputed_stats)
                    seq_list.append(Y_t)
                
                Y_list.append(torch.stack(seq_list, dim=0))
            
            if show_progress:
                print()
            return torch.stack(Y_list, dim=0)
        else:
            seq, N, F = X.shape
            
            # 原有模式：逐个切片处理
            Y_list = []
            for s_idx in range(seq):
                if show_progress and (s_idx + 1) == seq:
                    print(f"\r【预处理进度】处理切片: {s_idx+1}/{seq} ({(s_idx+1)*1000//seq}%)", end="", flush=True)
                
                # 对每个切片应用预处理（如果启用时序归一化，会在log后使用预计算统计量）
                Y_t = self.transform_slice_vectorized(X[s_idx], config, use_precomputed_stats=use_precomputed_stats)
                Y_list.append(Y_t)
            
            if show_progress:
                print()
            return torch.stack(Y_list, dim=0)
    
    def _apply_precomputed_temporal_norm(
        self,
        X_t: torch.Tensor,  # (N, F) log之后的切片
        device: torch.device,
        config: Dict,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        使用预计算的统计量应用时序归一化
        
        参数:
            X_t: (N, F) log之后的切片
            device: 设备
            config: 配置字典
            eps: 小常数
        
        返回:
            X_normalized: (N, F) 归一化后的切片
        """
        if self.stats is None:
            return X_t
        
        # 将统计量移到正确的设备
        stats_device = {}
        for key, value in self.stats.items():
            stats_device[key] = value.to(device)
        
        temporal_norm_cfg = config.get("temporal_norm", {})
        norm_method = temporal_norm_cfg.get("method", "zscore")  # zscore, minmax, std, range, median
        
        N, F = X_t.shape
        
        if norm_method == "zscore":
            # Z标准化: (x - mean) / std
            mean = stats_device['mean']  # (N, F)
            std = stats_device['std']    # (N, F)
            X_normalized = (X_t - mean) / (std + eps)
        
        elif norm_method == "minmax":
            # MinMax归一化: (x - min) / (max - min) * 2 - 1  (缩放到[-1, 1])
            min_val = stats_device['min']  # (N, F)
            max_val = stats_device['max']  # (N, F)
            ranges = max_val - min_val
            valid = ranges > eps
            # 对整个张量计算归一化结果
            X_normalized_full = 2.0 * (X_t - min_val) / (ranges + eps) - 1.0
            # 只对有效范围的位置应用归一化，其他位置保持为0
            X_normalized = torch.where(valid, X_normalized_full, torch.zeros_like(X_t))
            X_normalized = torch.clamp(X_normalized, -1.0, 1.0)
        
        elif norm_method == "std":
            # 除以标准差
            std = stats_device['std']  # (N, F)
            X_normalized = X_t / (std + eps)
        
        elif norm_method == "range":
            # 除以极差
            min_val = stats_device['min']  # (N, F)
            max_val = stats_device['max']  # (N, F)
            ranges = max_val - min_val
            ranges = torch.where(ranges < eps, torch.ones_like(ranges), ranges)
            X_normalized = X_t / (ranges + eps)
        
        elif norm_method == "median":
            # 除以中位数
            median = stats_device['median']  # (N, F)
            median_abs = torch.abs(median)
            median_abs = torch.where(median_abs < eps, torch.ones_like(median_abs), median_abs)
            X_normalized = X_t / (median_abs + eps)
        
        else:
            # 默认不归一化
            X_normalized = X_t
        
        # 处理nan和inf：重置回中位数
        median = stats_device['median']  # (N, F)
        invalid_mask = ~torch.isfinite(X_normalized)
        if invalid_mask.any():
            X_normalized = torch.where(invalid_mask, median, X_normalized)
        
        return X_normalized

    def transform_slice_vectorized(
        self, 
        X_t: torch.Tensor, 
        config: Dict,
        use_precomputed_stats: bool = False
    ) -> torch.Tensor:
        """
        单切片预处理方法（支持使用预计算统计量的时序归一化）
        
        参数:
            X_t: (N, F) 输入切片
            config: 配置字典
            use_precomputed_stats: 是否使用预计算统计量进行时序归一化
        """
        device = X_t.device
        N, F = X_t.shape
        eps = 1e-6
        x_f = X_t.clone()

        # 配置
        check_invalid_values = config.get("check_invalid_values", True)
        exclude_filled_nan_in_stats = config.get("exclude_filled_nan_in_stats", False) and check_invalid_values
        set_filled_nan_to_zero = config.get("set_filled_nan_to_zero", False) and check_invalid_values
        original_invalid_mask = None

        # 0. 统一异常值
        if check_invalid_values:
            original_invalid_mask = torch.isnan(x_f) | torch.isinf(x_f)
            x_f = torch.where(original_invalid_mask, torch.tensor(float('nan'), device=device, dtype=x_f.dtype), x_f)

        # 1. NaN 填充
        if config.get("enable_nan_fill", True) and check_invalid_values:
            median_vals = _nanmedian_vectorized(x_f, dim=0)
            median_vals = torch.where(torch.isnan(median_vals), torch.zeros_like(median_vals), median_vals)
            x_f = torch.where(original_invalid_mask, median_vals.unsqueeze(0), x_f)

        # 2. Log
        if config.get("enable_log", True):
            x_f = torch.sign(x_f) * torch.log1p(torch.abs(x_f))
            if check_invalid_values:
                new_invalid = ~torch.isfinite(x_f)
                if new_invalid.any():
                    original_invalid_mask = original_invalid_mask | new_invalid if original_invalid_mask is not None else new_invalid
                    if config.get("enable_nan_fill", True):
                        median_vals = _nanmedian_vectorized(x_f, dim=0)
                        median_vals = torch.where(torch.isnan(median_vals), torch.zeros_like(median_vals), median_vals)
                        x_f = torch.where(new_invalid, median_vals.unsqueeze(0), x_f)

        # 2.5. 时序归一化（使用预计算统计量，在log之后）
        if use_precomputed_stats:
            x_f = self._apply_precomputed_temporal_norm(x_f, device, config, eps)
            # 处理可能的新的nan/inf
            if check_invalid_values:
                new_invalid = ~torch.isfinite(x_f)
                if new_invalid.any():
                    original_invalid_mask = original_invalid_mask | new_invalid if original_invalid_mask is not None else new_invalid

        # 3. Winsor
        if config.get("enable_winsor", False):
            winsor_low = config.get("winsor_low", 0.01)
            winsor_high = config.get("winsor_high", 0.99)
            if exclude_filled_nan_in_stats and original_invalid_mask is not None:
                x_stats = torch.where(original_invalid_mask, torch.tensor(float('nan'), device=device, dtype=x_f.dtype), x_f)
                q_low = _nanquantile_vectorized(x_stats, torch.tensor(winsor_low, device=device), dim=0)
                q_high = _nanquantile_vectorized(x_stats, torch.tensor(winsor_high, device=device), dim=0)
            elif check_invalid_values:
                q_low = _nanquantile_vectorized(x_f, torch.tensor(winsor_low, device=device), dim=0)
                q_high = _nanquantile_vectorized(x_f, torch.tensor(winsor_high, device=device), dim=0)
            else:
                q_low = torch.quantile(x_f, winsor_low, dim=0)
                q_high = torch.quantile(x_f, winsor_high, dim=0)
            q_low = torch.where(torch.isnan(q_low), torch.zeros_like(q_low), q_low)
            q_high = torch.where(torch.isnan(q_high), torch.zeros_like(q_high), q_high)
            x_f = torch.clamp(x_f, q_low.unsqueeze(0), q_high.unsqueeze(0))

        # 4. Z-score step1
        if config.get("enable_zscore_step1", False):
            if exclude_filled_nan_in_stats and original_invalid_mask is not None:
                x_stats = torch.where(original_invalid_mask, torch.tensor(float('nan'), device=device, dtype=x_f.dtype), x_f)
                mu = _nanmean_vectorized(x_stats, dim=0)
                std = _nanstd_vectorized(x_stats, dim=0)
            elif check_invalid_values:
                mu = _nanmean_vectorized(x_f, dim=0)
                std = _nanstd_vectorized(x_f, dim=0)
            else:
                mu = x_f.mean(dim=0)
                std = x_f.std(dim=0)
            mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
            std = torch.where(torch.isnan(std) | (std < eps), torch.ones_like(std), std)
            x_f = (x_f - mu.unsqueeze(0)) / (std.unsqueeze(0) + eps)
            if check_invalid_values:
                new_invalid = ~torch.isfinite(x_f)
                if new_invalid.any():
                    original_invalid_mask = original_invalid_mask | new_invalid

        # 5. clip
        if config.get("enable_clip", False):
            clip_value = config.get("clip_value", 5.0)
            x_f = torch.clamp(x_f, -clip_value, clip_value)

        # ========= 融合 =========
        fused = self._apply_fusion(x_f, original_invalid_mask, config, device, eps)

        # 恢复异常位为0
        if set_filled_nan_to_zero and original_invalid_mask is not None:
            fused = torch.where(original_invalid_mask, torch.zeros_like(fused), fused)

        return fused
    
    def _apply_fusion(
        self,
        x_f: torch.Tensor,
        original_invalid_mask: Optional[torch.Tensor],
        config: Dict,
        device: torch.device,
        eps: float
    ) -> torch.Tensor:
        """
        应用融合逻辑（提取自transform_slice_vectorized）
        """
        check_invalid_values = config.get("check_invalid_values", True)
        exclude_filled_nan_in_stats = config.get("exclude_filled_nan_in_stats", False) and check_invalid_values
        
        fusion_config = config.get("fusion", {})
        use_minmax = fusion_config.get("use_minmax", False)
        use_zscore_step2 = fusion_config.get("use_zscore_step2", False)
        use_rank = fusion_config.get("use_rank", False)
        use_raw = fusion_config.get("use_raw", False)
        use_robust_z = fusion_config.get("use_robust_z", False)
        use_iqr_scale = fusion_config.get("use_iqr_scale", False)
        use_boxcox_yj = fusion_config.get("use_boxcox_yj", False)
        use_asinh_signedlog = fusion_config.get("use_asinh_signedlog", False)
        use_tanh_estimator = fusion_config.get("use_tanh_estimator", False)

        weight_minmax = fusion_config.get("weight_minmax", 1.0)
        weight_zscore = fusion_config.get("weight_zscore", 1.0)
        weight_rank = fusion_config.get("weight_rank", 1.0)
        weight_raw = fusion_config.get("weight_raw", 1.0)
        weight_robust_z = fusion_config.get("weight_robust_z", 1.0)
        weight_iqr_scale = fusion_config.get("weight_iqr_scale", 1.0)
        weight_boxcox_yj = fusion_config.get("weight_boxcox_yj", 1.0)
        weight_asinh_signedlog = fusion_config.get("weight_asinh_signedlog", 1.0)
        weight_tanh_estimator = fusion_config.get("weight_tanh_estimator", 1.0)

        total_weight = 0.0
        for flag, w in [
            (use_minmax, weight_minmax),
            (use_zscore_step2, weight_zscore),
            (use_rank, weight_rank),
            (use_raw, weight_raw),
            (use_robust_z, weight_robust_z),
            (use_iqr_scale, weight_iqr_scale),
            (use_boxcox_yj, weight_boxcox_yj),
            (use_asinh_signedlog, weight_asinh_signedlog),
            (use_tanh_estimator, weight_tanh_estimator),
        ]:
            if flag:
                total_weight += w
        if total_weight < 1e-6:
            use_raw = True
            weight_raw = 1.0
            total_weight = 1.0

        channels = []
        weights = []

        # 预计算统计输入
        if check_invalid_values:
            if exclude_filled_nan_in_stats and original_invalid_mask is not None:
                stats_valid_mask = ~original_invalid_mask
                x_f_for_stats = torch.where(original_invalid_mask, torch.tensor(float('nan'), device=device, dtype=x_f.dtype), x_f)
            else:
                stats_valid_mask = torch.isfinite(x_f)
                x_f_for_stats = x_f
        else:
            stats_valid_mask = None
            x_f_for_stats = x_f

        # 1. Min-Max
        if use_minmax:
            cfg = fusion_config.get("minmax", {})
            ql = cfg.get("quantile_low", 0.01)
            qh = cfg.get("quantile_high", 0.99)
            q_low = _nanquantile_vectorized(x_f_for_stats, torch.tensor(ql, device=device, dtype=x_f.dtype), dim=0) if check_invalid_values else torch.quantile(x_f, ql, dim=0)
            q_high = _nanquantile_vectorized(x_f_for_stats, torch.tensor(qh, device=device, dtype=x_f.dtype), dim=0) if check_invalid_values else torch.quantile(x_f, qh, dim=0)
            q_low = torch.where(torch.isnan(q_low), torch.zeros_like(q_low), q_low)
            q_high = torch.where(torch.isnan(q_high), torch.zeros_like(q_high), q_high)
            ranges = q_high - q_low
            valid = ranges > eps  # (F,)
            # 对整个张量计算归一化结果
            # q_low 和 q_high 需要广播到 (N, F)
            mm_channel_full = 2.0 * (x_f - q_low.unsqueeze(0)) / (ranges.unsqueeze(0) + eps) - 1.0
            # 只对有效范围的位置应用归一化，其他位置保持为0
            # valid 需要广播到 (N, F)
            mm_channel = torch.where(valid.unsqueeze(0), mm_channel_full, torch.zeros_like(x_f))
            mm_channel = torch.clamp(mm_channel, -1.0, 1.0)
            channels.append(mm_channel)
            weights.append(weight_minmax)

        # 2. Z-score step2
        if use_zscore_step2:
            mu = _nanmean_vectorized(x_f_for_stats, dim=0) if check_invalid_values else x_f.mean(dim=0)
            std = _nanstd_vectorized(x_f_for_stats, dim=0) if check_invalid_values else x_f.std(dim=0)
            mu = torch.where(torch.isnan(mu), torch.zeros_like(mu), mu)
            std = torch.where(torch.isnan(std) | (std < eps), torch.ones_like(std), std)
            z_channel = (x_f - mu.unsqueeze(0)) / (std.unsqueeze(0) + eps)
            z_channel = torch.tanh(z_channel)
            channels.append(z_channel)
            weights.append(weight_zscore)

        # 3. Rank
        if use_rank:
            rank_method = fusion_config.get("rank_method", "uniform")
            N = x_f.shape[0]
            if rank_method == "adaptive":
                ad = fusion_config.get("rank_adaptive", {})
                head_ratio = ad.get("head_ratio", 0.35)
                tail_ratio = ad.get("tail_ratio", 0.35)
                head_buckets = ad.get("head_buckets", 2000)
                tail_buckets = ad.get("tail_buckets", 2000)
                mid_buckets = ad.get("mid_buckets", 500)
                quantiles = _compute_adaptive_quantiles(head_ratio, tail_ratio, head_buckets, mid_buckets, tail_buckets, device)
            else:
                num_buckets = min(N, 10000)
                quantiles = torch.linspace(0, 1, num_buckets + 1, device=device)
            quantile_values = _nanquantile_vectorized(x_f_for_stats, quantiles, dim=0) if check_invalid_values else torch.quantile(x_f, quantiles, dim=0)
            quantile_values = torch.where(torch.isnan(quantile_values), torch.zeros_like(quantile_values), quantile_values)
            x_f_T = x_f.T.contiguous()
            q_T = quantile_values.T.contiguous()
            rank_buckets = torch.searchsorted(q_T, x_f_T, right=True).T
            if rank_method == "adaptive":
                rank_norm = _bucket_to_rank_adaptive(rank_buckets, quantiles)
            else:
                total_buckets = quantiles.numel() - 1
                rank_norm = (rank_buckets.float() - 0.5) / total_buckets
            rank_channel = 2.0 * rank_norm - 1.0
            rank_channel = torch.clamp(rank_channel, -1.0, 1.0)
            channels.append(rank_channel)
            weights.append(weight_rank)

        # 4. Raw
        if use_raw:
            channels.append(x_f.clone())
            weights.append(weight_raw)

        # 5. Robust-Z（MAD）
        if use_robust_z:
            if check_invalid_values:
                median_vals = _nanmedian_vectorized(x_f_for_stats, dim=0)
            else:
                median_vals = x_f.median(dim=0)[0]
            median_vals = torch.where(torch.isnan(median_vals), torch.zeros_like(median_vals), median_vals)
            
            if check_invalid_values:
                mad_vals = _nanmedian_vectorized(torch.abs(x_f_for_stats - median_vals.unsqueeze(0)), dim=0)
            else:
                mad_vals = torch.abs(x_f - median_vals.unsqueeze(0)).median(dim=0)[0]
            mad_vals = torch.where(torch.isnan(mad_vals) | (mad_vals < eps), torch.ones_like(mad_vals), mad_vals)
            
            robust_z = (x_f - median_vals.unsqueeze(0)) / (mad_vals.unsqueeze(0) + eps)
            robust_z = torch.where(~torch.isfinite(robust_z), torch.zeros_like(robust_z), robust_z)
            robust_z = torch.tanh(robust_z)
            channels.append(robust_z)
            weights.append(weight_robust_z)

        # 6. IQR-Scale
        if use_iqr_scale:
            if check_invalid_values:
                q25 = _nanquantile_vectorized(x_f_for_stats, torch.tensor(0.25, device=device, dtype=x_f.dtype), dim=0)
                q50 = _nanquantile_vectorized(x_f_for_stats, torch.tensor(0.50, device=device, dtype=x_f.dtype), dim=0)
                q75 = _nanquantile_vectorized(x_f_for_stats, torch.tensor(0.75, device=device, dtype=x_f.dtype), dim=0)
            else:
                q25 = torch.quantile(x_f, torch.tensor(0.25, device=device, dtype=x_f.dtype), dim=0)
                q50 = torch.quantile(x_f, torch.tensor(0.50, device=device, dtype=x_f.dtype), dim=0)
                q75 = torch.quantile(x_f, torch.tensor(0.75, device=device, dtype=x_f.dtype), dim=0)
            
            q25 = torch.where(torch.isnan(q25), torch.zeros_like(q25), q25)
            q50 = torch.where(torch.isnan(q50), torch.zeros_like(q50), q50)
            q75 = torch.where(torch.isnan(q75), torch.zeros_like(q75), q75)
            
            iqr = q75 - q25
            iqr = torch.where(torch.isnan(iqr) | (iqr < eps), torch.ones_like(iqr), iqr)
            
            iqr_scaled = (x_f - q50.unsqueeze(0)) / (iqr.unsqueeze(0) + eps)
            iqr_scaled = torch.where(~torch.isfinite(iqr_scaled), torch.zeros_like(iqr_scaled), iqr_scaled)
            iqr_scaled = torch.tanh(iqr_scaled)
            channels.append(iqr_scaled)
            weights.append(weight_iqr_scale)

        # 7. Box-Cox / Yeo-Johnson
        if use_boxcox_yj:
            bc_cfg = fusion_config.get("boxcox_yj", {})
            lam = torch.tensor(bc_cfg.get("lambda", 0.0), device=device, dtype=x_f.dtype)
            method = bc_cfg.get("method", "yeo_johnson")
            shift = bc_cfg.get("shift", 1e-6)
            if method == "boxcox":
                x_bc = torch.clamp(x_f + shift, min=shift)
                if torch.abs(lam) < 1e-6:
                    bc_channel = torch.log(x_bc)
                else:
                    bc_channel = (torch.pow(x_bc, lam) - 1.0) / lam
            else:  # yeo_johnson
                pos = x_f >= 0
                x_pos = x_f[pos] + shift
                x_neg = -x_f[~pos] + shift
                yj = torch.zeros_like(x_f)
                if torch.abs(lam) < 1e-6:
                    yj[pos] = torch.log(x_pos)
                    yj[~pos] = -torch.log(x_neg)
                else:
                    yj[pos] = (torch.pow(x_pos, lam) - 1.0) / lam
                    yj[~pos] = -((torch.pow(x_neg, 2 - lam) - 1.0) / (2 - lam))
                bc_channel = yj
            bc_channel = torch.where(~torch.isfinite(bc_channel), torch.zeros_like(bc_channel), bc_channel)
            channels.append(bc_channel)
            weights.append(weight_boxcox_yj)

        # 8. asinh / signed-log
        if use_asinh_signedlog:
            asl_cfg = fusion_config.get("asinh_signedlog", {})
            mode = asl_cfg.get("mode", "asinh")
            scale_from = asl_cfg.get("scale_from", "mad")
            scale_mul = asl_cfg.get("scale_mul", 1.0)
            if scale_from == "iqr":
                if check_invalid_values:
                    q25 = _nanquantile_vectorized(x_f_for_stats, torch.tensor(0.25, device=device, dtype=x_f.dtype), dim=0)
                    q75 = _nanquantile_vectorized(x_f_for_stats, torch.tensor(0.75, device=device, dtype=x_f.dtype), dim=0)
                else:
                    q25 = torch.quantile(x_f, torch.tensor(0.25, device=device, dtype=x_f.dtype), dim=0)
                    q75 = torch.quantile(x_f, torch.tensor(0.75, device=device, dtype=x_f.dtype), dim=0)
                scale = q75 - q25
            else:
                if check_invalid_values:
                    median_vals = _nanmedian_vectorized(x_f_for_stats, dim=0)
                    scale = _nanmedian_vectorized(torch.abs(x_f_for_stats - median_vals.unsqueeze(0)), dim=0)
                else:
                    median_vals = x_f.median(dim=0)[0]
                    scale = torch.abs(x_f - median_vals.unsqueeze(0)).median(dim=0)[0]
            scale = torch.where(torch.isnan(scale) | (scale < eps), torch.ones_like(scale), scale) * scale_mul
            if mode == "signed_log":
                asl_channel = torch.sign(x_f) * torch.log1p(torch.abs(x_f) / scale.unsqueeze(0))
            else:
                asl_channel = torch.asinh(x_f / scale.unsqueeze(0))
            asl_channel = torch.where(~torch.isfinite(asl_channel), torch.zeros_like(asl_channel), asl_channel)
            channels.append(asl_channel)
            weights.append(weight_asinh_signedlog)

        # 9. Tanh-Estimator
        if use_tanh_estimator:
            te_cfg = fusion_config.get("tanh_estimator", {})
            kappa = te_cfg.get("kappa", 0.8)
            if check_invalid_values:
                median_vals = _nanmedian_vectorized(x_f_for_stats, dim=0)
                mad_vals = _nanmedian_vectorized(torch.abs(x_f_for_stats - median_vals.unsqueeze(0)), dim=0)
            else:
                median_vals = x_f.median(dim=0)[0]
                mad_vals = torch.abs(x_f - median_vals.unsqueeze(0)).median(dim=0)[0]
            median_vals = torch.where(torch.isnan(median_vals), torch.zeros_like(median_vals), median_vals)
            mad_vals = torch.where(torch.isnan(mad_vals) | (mad_vals < eps), torch.ones_like(mad_vals), mad_vals)
            te_channel = torch.tanh((x_f - median_vals.unsqueeze(0)) / (kappa * mad_vals.unsqueeze(0) + eps))
            te_channel = torch.where(~torch.isfinite(te_channel), torch.zeros_like(te_channel), te_channel)
            channels.append(te_channel)
            weights.append(weight_tanh_estimator)

        # 加权融合
        if len(channels) == 1:
            fused = channels[0]
        else:
            channels_tensor = torch.stack(channels, dim=0)
            weights_tensor = torch.tensor(weights, device=device, dtype=x_f.dtype) / total_weight
            fused = (channels_tensor * weights_tensor.view(-1, 1, 1)).sum(dim=0)
        
        return fused

 