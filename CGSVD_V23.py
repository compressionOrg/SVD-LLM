#coding:utf8
import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import contextlib
import io
import sys

from utils.data_utils import get_calib_train_data
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import get_model_from_huggingface, find_layers
from evaluater import *

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class LayerSVDInfo:
    """存储每一层每个线性层的 SVD 分解信息"""
    layer_idx: int
    linear_name: str
    U: torch.Tensor           # 左奇异向量
    S: torch.Tensor           # 奇异值 (降序排列)
    VT: torch.Tensor          # 右奇异向量
    scaling_matrix: torch.Tensor     # 白化变换矩阵
    scaling_matrix_inv: torch.Tensor # 白化逆变换矩阵
    rows: int                 # 权重矩阵行数 (输出维度)
    cols: int                 # 权重矩阵列数 (输入维度)


@dataclass
class SpectralFeature:
    """存储每一层的谱特征信息"""
    layer_idx: int
    linear_name: str
    spectral_entropy: float   # 谱熵 (基于奇异值分布)
    max_rank: int             # 最大可能秩
    rows: int                 # 权重矩阵行数
    cols: int                 # 权重矩阵列数


# ============================================================================
# 阶段一：数据感知白化 (Whitening) - 空间重构
# ============================================================================

@torch.no_grad()
def compute_whitening_matrices(model_name: str, model, calib_loader, dev: str) -> Dict:
    """
    计算白化变换矩阵
    
    原理: 通过输入协方差矩阵对权重进行变换，将"权重重要"转化为"激活重要"
    
    Returns:
        profiling_mat: 包含每层每个线性层的白化矩阵
    """
    print("=" * 60)
    print("[阶段一] 计算数据感知白化矩阵...")
    print("=" * 60)
    
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 
             'cache_position': None, 'position_embeddings': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                attn = kwargs.get('attention_mask', None)
                cache['attention_mask'] = attn.cpu() if attn is not None else None
                if "opt" not in model_name:
                    pos_ids = kwargs.get('position_ids', None)
                    cache['position_ids'] = pos_ids.cpu() if pos_ids is not None else None
                    cache_pos = kwargs.get('cache_position', None)
                    cache['cache_position'] = cache_pos.cpu() if cache_pos is not None else None
                    pos_embs = kwargs.get('position_embeddings', None)
                    if pos_embs is not None:
                        cos, sin = pos_embs
                        cache['position_embeddings'] = (cos.cpu(), sin.cpu())
            else:
                attn = kwargs.get('attention_mask', None)
                if attn is not None:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], attn.cpu()), dim=0)
                if "opt" not in model_name:
                    pos_ids = kwargs.get('position_ids', None)
                    if pos_ids is not None:
                        cache['position_ids'] = torch.cat((cache['position_ids'], pos_ids.cpu()), dim=0)
                    cache_pos = kwargs.get('cache_position', None)
                    if cache_pos is not None:
                        cache['cache_position'] = torch.cat((cache['cache_position'], cache_pos.cpu()), dim=0)
                    pos_embs = kwargs.get('position_embeddings', None)
                    if pos_embs is not None:
                        cos, sin = pos_embs
                        cached = cache['position_embeddings']
                        if cached is None:
                            cache['position_embeddings'] = (cos.cpu(), sin.cpu())
                        else:
                            cache['position_embeddings'] = (
                                torch.cat((cached[0], cos.cpu()), dim=0),
                                torch.cat((cached[1], sin.cpu()), dim=0),
                            )
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            
            # Check for input validity
            if 'input_ids' in batch:
                input_ids = batch['input_ids']
                vocab_size = model.config.vocab_size
                if input_ids.max() >= vocab_size:
                    print(f"Error: input_ids max value ({input_ids.max()}) exceeds vocab size ({vocab_size})")
                    # Clip or handle error
                    # input_ids[input_ids >= vocab_size] = vocab_size - 1
                    raise ValueError(f"Input IDs exceed vocabulary size: {input_ids.max()} >= {vocab_size}")
                if input_ids.min() < 0:
                    print(f"Error: input_ids contains negative values ({input_ids.min()})")
                    raise ValueError(f"Input IDs contain negative values: {input_ids.min()}")
            
            model(**batch)
        except ValueError:
            pass
        except RuntimeError as e:
            print(f"RuntimeError during calibration: {e}")
            raise e
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
        cache_position = cache['cache_position']
        position_embeddings = cache['position_embeddings']
    
    profiling_mat = {}
    
    print("逐层计算白化矩阵...")
    for i in tqdm(range(len(layers)), desc="White Layer"):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                kwargs = {}
                if attention_masks is not None:
                    kwargs['attention_mask'] = attention_masks[j].unsqueeze(0).to(dev)
                if position_ids is not None:
                    if position_ids.shape[0] == 1:
                        kwargs['position_ids'] = position_ids.to(dev)
                    else:
                        kwargs['position_ids'] = position_ids[j].unsqueeze(0).to(dev)
                if cache_position is not None:
                    if cache_position.shape[0] == 1:
                        kwargs['cache_position'] = cache_position.to(dev)
                    else:
                        kwargs['cache_position'] = cache_position[j].unsqueeze(0).to(dev)
                if position_embeddings is not None:
                    if position_embeddings[0].shape[0] == 1:
                        cos_j = position_embeddings[0].to(dev)
                        sin_j = position_embeddings[1].to(dev)
                    else:
                        cos_j = position_embeddings[0][j].unsqueeze(0).to(dev)
                        sin_j = position_embeddings[1][j].unsqueeze(0).to(dev)
                    kwargs['position_embeddings'] = (cos_j, sin_j)
                outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
            else:
                if attention_masks is not None:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_masks[j].unsqueeze(0).to(dev),
                    )[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
        
        for h in handles:
            h.remove()
        
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 3e-4) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            del scaling_diag_matrix, raw_scaling_diag_matrix
            if hasattr(subset[name], 'scaling_diag_matrix'):
                del subset[name].scaling_diag_matrix
            torch.cuda.empty_cache()
        
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs.clone()
        torch.cuda.empty_cache()
    
    print(f"✓ 白化矩阵计算完成，共 {len(layers)} 层")
    return profiling_mat


# ============================================================================
# 阶段二：SVD 分解
# ============================================================================

@torch.no_grad()
def compute_svd_and_importance(
    model_name: str, 
    model, 
    profiling_mat: Dict,
    dev: str,
    save_path: str = None
) -> Dict[Tuple[int, str], LayerSVDInfo]:
    """
    对所有权重矩阵在白化空间进行 SVD 分解
    
    Returns:
        svd_info: 字典 {(layer_idx, linear_name): LayerSVDInfo}
    """
    print("\n" + "=" * 60)
    print("[阶段二-准备] 在白化空间进行 SVD 分解...")
    print("=" * 60)
    
    # 缓存目录
    if save_path:
        cache_dir = save_path
    else:
        cache_dir = "svd_cache"
        
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # 构建缓存文件名 (处理路径分隔符)
    safe_model_name = model_name.replace("/", "_")
    cache_file = os.path.join(cache_dir, f"svd_info_{safe_model_name}.pt")
    
    # 尝试加载缓存
    if os.path.exists(cache_file):
        print(f"检测到缓存文件 {cache_file}，正在加载...")
        try:
            svd_info = torch.load(cache_file, map_location="cpu")
            print(f"✓ 成功加载缓存 SVD 结果，共 {len(svd_info)} 个权重矩阵")
            return svd_info
        except Exception as e:
            print(f"加载缓存失败: {e}，将重新计算...")
    
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    
    svd_info = {}
    
    for i in tqdm(range(len(layers)), desc="SVD Layer"):
        layer = layers[i]
        subset = find_layers(layer)
        
        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except:
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            
            svd_info[(i, name)] = LayerSVDInfo(
                layer_idx=i,
                linear_name=name,
                U=U.cpu(),
                S=S.cpu(),
                VT=VT.cpu(),
                scaling_matrix=scaling_diag_matrix.cpu(),
                scaling_matrix_inv=scaling_matrix_inv.cpu(),
                rows=W.shape[0],
                cols=W.shape[1]
            )
            
            del W, W_scale, U, S, VT, scaling_diag_matrix, scaling_matrix_inv
            torch.cuda.empty_cache()
    
    # 保存缓存
    # print(f"正在保存 SVD 结果到 {cache_file}...")
    # torch.save(svd_info, cache_file)
    # print("✓ 缓存保存完成")

    print(f"✓ SVD 分解完成，共 {len(svd_info)} 个权重矩阵")
    return svd_info


def print_rank_statistics(
    svd_info: Dict[Tuple[int, str], LayerSVDInfo],
    rank_allocation: Dict[Tuple[int, str], int]
):
    """打印秩分配统计信息"""
    print("\n--- 秩分配统计 ---")
    
    layer_stats = {}
    for key, rank in rank_allocation.items():
        layer_idx, linear_name = key
        info = svd_info[key]
        max_rank = min(info.rows, info.cols)
        rank_ratio = rank / max_rank
        
        if layer_idx not in layer_stats:
            layer_stats[layer_idx] = []
        layer_stats[layer_idx].append((linear_name, rank, max_rank, rank_ratio))
    
    for layer_idx in sorted(layer_stats.keys())[:3]:
        stats = layer_stats[layer_idx]
        avg_ratio = np.mean([s[3] for s in stats])
        print(f"Layer {layer_idx}: 平均秩比例 {avg_ratio:.2%}")
        for name, rank, max_rank, ratio in stats:
            print(f"  {name}: {rank}/{max_rank} ({ratio:.2%})")
    
    if len(layer_stats) > 3:
        print(f"  ... (共 {len(layer_stats)} 层)")
    
    all_ratios = [rank_allocation[k] / min(svd_info[k].rows, svd_info[k].cols) 
                  for k in rank_allocation]
    print(f"\n全模型秩比例: min={min(all_ratios):.2%}, max={max(all_ratios):.2%}, mean={np.mean(all_ratios):.2%}")


# ============================================================================
# 阶段二：基于 spectral_entropy 的分层秩分配
# ============================================================================

def compute_entropy(singular_values: torch.Tensor) -> float:
    """
    计算香农熵 (Spectral Entropy)
    
    公式: H = - sum(p * log(p))
    其中 p 为归一化奇异值能量分布
    
    Args:
        singular_values: 奇异值序列
        
    Returns:
        entropy: 香农熵 (数值越大，分布越均匀，重要性越高)
    """
    S = singular_values.cpu().numpy()
    energy = S ** 2
    total_energy = np.sum(energy)
    
    if total_energy < 1e-10:
        return 0.0  # 能量极小，视为无信息，熵为0
    
    # 归一化为概率分布 p_i
    p = energy / total_energy
    
    # 为了数值稳定性，只对 p > 0 的项计算 log
    p = p[p > 1e-20]
    
    # 计算香农熵 H = -sum(p * log(p))
    entropy = -np.sum(p * np.log(p))
    
    return float(entropy)


@torch.no_grad()
def extract_spectral_features(
    svd_info: Dict[Tuple[int, str], LayerSVDInfo]
) -> Dict[Tuple[int, str], SpectralFeature]:
    """
    从 SVD 信息中提取谱特征 (仅计算 spectral_entropy)
    
    Args:
        svd_info: SVD 分解信息
        
    Returns:
        spectral_features: {(layer_idx, linear_name): SpectralFeature}
    """
    print("\n" + "=" * 60)
    print("[阶段二-谱分析] 计算谱熵 (spectral_entropy)...")
    print("=" * 60)
    
    spectral_features = {}
    entropies = []
    
    for key, info in tqdm(svd_info.items(), desc="Computing Spectral Entropy"):
        layer_idx, linear_name = key
        
        entropy = compute_entropy(info.S)
        max_rank = min(info.rows, info.cols)
        
        spectral_features[key] = SpectralFeature(
            layer_idx=layer_idx,
            linear_name=linear_name,
            spectral_entropy=entropy,
            max_rank=max_rank,
            rows=info.rows,
            cols=info.cols
        )
        
        entropies.append(entropy)
    
    print(f"\n--- 谱熵统计 ---")
    print(f"spectral_entropy: min={min(entropies):.4f}, max={max(entropies):.4f}, mean={np.mean(entropies):.4f}")
    
    return spectral_features


@torch.no_grad()
def compute_layer_sensitivity(
    model_name: str, 
    model, 
    calib_loader, 
    dev: str, 
    metric: str = "cosine"
) -> Dict[int, float]:
    """
    计算每层的灵敏度/重要性 (Block Influence)
    
    Modes:
    - cosine: 1 - CosineSimilarity (computed over all tokens)
    - angular: Arccos(CosineSimilarity) / pi (computed over last token only, ref: ShortGPT)
    """
    print("\n" + "=" * 60)
    print(f"[阶段一补充] 计算层级灵敏度 (Metric: {metric})...")
    print("=" * 60)

    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 
             'cache_position': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                attn = kwargs.get('attention_mask', None)
                cache['attention_mask'] = attn.cpu() if attn is not None else None
                if "opt" not in model_name:
                    pos_ids = kwargs.get('position_ids', None)
                    cache['position_ids'] = pos_ids.cpu() if pos_ids is not None else None
                    cache_pos = kwargs.get('cache_position', None)
                    cache['cache_position'] = cache_pos.cpu() if cache_pos is not None else None
                    pos_embs = kwargs.get('position_embeddings', None)
                    if pos_embs is not None:
                        cos, sin = pos_embs
                        cache['position_embeddings'] = (cos.cpu(), sin.cpu())
            else:
                attn = kwargs.get('attention_mask', None)
                if attn is not None:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], attn.cpu()), dim=0)
                if "opt" not in model_name:
                    pos_ids = kwargs.get('position_ids', None)
                    if pos_ids is not None:
                        cache['position_ids'] = torch.cat((cache['position_ids'], pos_ids.cpu()), dim=0)
                    cache_pos = kwargs.get('cache_position', None)
                    if cache_pos is not None:
                        cache['cache_position'] = torch.cat((cache['cache_position'], cache_pos.cpu()), dim=0)
                    pos_embs = kwargs.get('position_embeddings', None)
                    if pos_embs is not None:
                        cos, sin = pos_embs
                        cached = cache['position_embeddings']
                        if cached is None:
                            cache['position_embeddings'] = (cos.cpu(), sin.cpu())
                        else:
                            cache['position_embeddings'] = (
                                torch.cat((cached[0], cos.cpu()), dim=0),
                                torch.cat((cached[1], sin.cpu()), dim=0),
                            )
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            # 检查 input_ids 是否在词表范围内
            if 'input_ids' in batch:
                vocab_size = model.config.vocab_size
                if (batch['input_ids'] >= vocab_size).any() or (batch['input_ids'] < 0).any():
                    invalid_indices = batch['input_ids'][(batch['input_ids'] >= vocab_size) | (batch['input_ids'] < 0)]
                    print(f"Error: input_ids contains invalid indices! Max valid index: {vocab_size - 1}")
                    print(f"Invalid indices found: {invalid_indices[:10]}...")
                    # 抛出 ValueError 以便被 except 块捕获并跳过此 batch
                    raise ValueError(f"Input IDs exceed vocabulary size or are negative.")
            
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
        cache_position = cache['cache_position']
        position_embeddings = cache['position_embeddings']
    
    layer_sensitivity = {}
    
    print(f"逐层计算灵敏度 (Metric: {metric})...")
    for i in tqdm(range(len(layers)), desc="Sensitivity"):
        layer = layers[i].to(dev)
        
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                kwargs = {}
                if attention_masks is not None:
                    kwargs['attention_mask'] = attention_masks[j].unsqueeze(0).to(dev)
                if position_ids is not None:
                    if position_ids.shape[0] == 1:
                        kwargs['position_ids'] = position_ids.to(dev)
                    else:
                        kwargs['position_ids'] = position_ids[j].unsqueeze(0).to(dev)
                if cache_position is not None:
                    if cache_position.shape[0] == 1:
                        kwargs['cache_position'] = cache_position.to(dev)
                    else:
                        kwargs['cache_position'] = cache_position[j].unsqueeze(0).to(dev)
                if position_embeddings is not None:
                    if position_embeddings[0].shape[0] == 1:
                        cos_j = position_embeddings[0].to(dev)
                        sin_j = position_embeddings[1].to(dev)
                    else:
                        cos_j = position_embeddings[0][j].unsqueeze(0).to(dev)
                        sin_j = position_embeddings[1][j].unsqueeze(0).to(dev)
                    kwargs['position_embeddings'] = (cos_j, sin_j)
                outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
            else:
                if attention_masks is not None:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_masks[j].unsqueeze(0).to(dev),
                    )[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0))[0]
        
        # 计算该层输入输出的差异/重要性
        if metric == 'angular':
            # ShortGPT Policy: Use Last Token Only for Angular
            # inps, outs: [batch, seqlen, hidden] -> [batch, hidden]
            inp_last = inps[:, -1, :]
            out_last = outs[:, -1, :]
            
            cos_sim = torch.nn.functional.cosine_similarity(inp_last, out_last, dim=-1) # [batch]
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            # Angular Distance: arccos(sim) / pi
            bi_score = (torch.acos(cos_sim) / torch.pi).mean().item()
            
        else: # cosine
            # Original Policy: Use All Tokens
            cos_sim = torch.nn.functional.cosine_similarity(inps, outs, dim=-1) # [batch, seqlen]
            avg_sim = cos_sim.mean().item()
            # Cosine Distance: 1 - sim
            bi_score = 1.0 - avg_sim
        
        layer_sensitivity[i] = bi_score
        
        layers[i] = layer.cpu()
        inps = outs.clone()
        torch.cuda.empty_cache()
    
    print(f"✓ 灵敏度计算完成")
    return layer_sensitivity


def compute_layer_importance(
    svd_info: Dict[Tuple[int, str], LayerSVDInfo],
    spectral_features: Dict[Tuple[int, str], SpectralFeature],
    layer_sensitivity: Dict[int, float] = None,
    temperature: float = 0.5
) -> Dict[int, float]:
    """
    计算每层的重要性分数
    
    Args:
        svd_info: SVD 分解信息
        spectral_features: 谱特征信息
        layer_sensitivity: 层级灵敏度 {layer_idx: sensitivity_score}
        temperature: Softmax 温度系数
    
    Returns:
        layer_importance: {layer_idx: importance_score}
    """
    layer_importance = {}
    
    if layer_sensitivity is not None:
        # print("Using Layer Sensitivity for Layer Importance...")
        
        # 去掉平滑，直接以 layer_sensitivity 设置为 layer_importance
        for layer_idx, sensitivity in layer_sensitivity.items():
            layer_importance[layer_idx] = sensitivity
            
    else:
        # Fallback to spectral entropy if no sensitivity provided
        print("Warning: No layer sensitivity provided, falling back to spectral entropy...")
        layer_info = {}
        for key, info in svd_info.items():
            layer_idx, linear_name = key
            if layer_idx not in layer_info:
                layer_info[layer_idx] = {'entropies': []}
            
            if key in spectral_features:
                sf = spectral_features[key]
                layer_info[layer_idx]['entropies'].append(sf.spectral_entropy)
        
        for layer_idx, info in layer_info.items():
            if info['entropies']:
                avg_entropy = np.mean(info['entropies'])
                layer_importance[layer_idx] = avg_entropy
            else:
                layer_importance[layer_idx] = 1.0
    
    total = sum(layer_importance.values())
    layer_importance = {k: v / total for k, v in layer_importance.items()}
    
    return layer_importance


def compute_intra_layer_importance(
    svd_info: Dict[Tuple[int, str], LayerSVDInfo],
    layer_idx: int,
    spectral_features: Dict[Tuple[int, str], SpectralFeature]
) -> Dict[str, float]:
    """
    计算层内各模块的重要性分数 (基于谱熵)
    
    Args:
        svd_info: SVD 分解信息
        layer_idx: 层索引
        spectral_features: 谱特征信息
    
    Returns:
        module_importance: {linear_name: importance_score}
    """
    layer_modules = {}
    for key, info in svd_info.items():
        if key[0] == layer_idx:
            linear_name = key[1]
            layer_modules[linear_name] = {'info': info}
            if key in spectral_features:
                sf = spectral_features[key]
                layer_modules[linear_name]['spectral_entropy'] = sf.spectral_entropy
    
    module_importance = {}
    for name, m in layer_modules.items():
        if 'spectral_entropy' in m:
            module_importance[name] = m['spectral_entropy']
        else:
            module_importance[name] = 1.0
    
    total = sum(module_importance.values())
    if total > 1e-10:
        module_importance = {k: v / total for k, v in module_importance.items()}
    else:
        # 如果总和为0 (例如所有熵都为0)，则均匀分配
        n_modules = len(module_importance)
        val = 1.0 / n_modules if n_modules > 0 else 0.0
        module_importance = {k: val for k in module_importance}
    
    return module_importance


    # 第一级：计算每层的参数预算
    print("\n--- 第一级: 层级参数预算分配 ---")
    
    pass 


def auto_determine_fluctuation_ratio(
    importances: np.ndarray,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    safety_factor: float = 0.95
) -> float:
    """
    自动计算最大的安全波动系数
    
    Args:
        importances: 原始重要性分数数组
        base_ratio: 目标基准比例 (Target Ratio)
        min_ratio: 允许的最小比例
        max_ratio: 允许的最大比例
        safety_factor: 安全因子，保留一点余量 (默认0.95)
        
    Returns:
        fluctuation_ratio: 推荐的波动系数
    """
    # 1. 预处理重要性 (归一化 + 去均值)
    if len(importances) < 2 or (importances.max() == importances.min()):
        return 0.0
        
    # Min-Max 归一化到 [0, 1]
    imp_norm = (importances - importances.min()) / (importances.max() - importances.min())
    # 去均值
    imp_centered = imp_norm - np.mean(imp_norm)
    
    # 2. 计算每个点允许的最大波动系数
    # 公式: Ratio = Base + k * Centered_Imp
    # => k * Centered_Imp <= Max - Base  (当 Imp > 0)
    # => k * Centered_Imp >= Min - Base  (当 Imp < 0)
    
    max_k = float('inf')
    
    for val in imp_centered:
        if abs(val) < 1e-6:
            continue
            
        if val > 0:
            # 正向偏离，受 Max 限制
            allowed_k = (max_ratio - base_ratio) / val
        else:
            # 负向偏离，受 Min 限制
            # val is negative, so division flips sign behavior. 
            # We want k * val >= min - base => k <= (min - base) / val
            # => k <= (base - min) / abs(val)
            allowed_k = (base_ratio - min_ratio) / abs(val)
            
        if allowed_k < max_k:
            max_k = allowed_k
            
    # 3. 应用安全因子并返回
    if max_k == float('inf'):
        return 0.0
        
    return max_k * safety_factor


@torch.no_grad()
def hierarchical_rank_allocation(
    svd_info: Dict[Tuple[int, str], LayerSVDInfo],
    target_param_ratio: float,
    min_rank_ratio: float = 0.05,
    max_rank_ratio: float = 0.95,
    spectral_features: Dict[Tuple[int, str], SpectralFeature] = None,
    layer_sensitivity: Dict[int, float] = None,
    temperature: float = 0.5,
    layer_fluctuation_ratio: float = 0.5,
    module_fluctuation_ratio: float = 0.5
) -> Dict[Tuple[int, str], int]:
    """
    分层秩分配算法 (Hierarchical Rank Allocation)
    
    两级分配策略:
    1. 第一级：计算每层 layer 的参数预算（基于层级灵敏度）
    2. 第二级：在层内按模块分配秩（基于模块谱熵 spectral_entropy）
    
    Args:
        svd_info: SVD 分解信息
        target_param_ratio: 目标参数保留比例
        min_rank_ratio: 最小秩比例
        max_rank_ratio: 最大秩比例
        spectral_features: 谱特征信息
        layer_sensitivity: 层级灵敏度
        temperature: 平滑温度系数
    
    Returns:
        rank_allocation: {(layer_idx, linear_name): allocated_rank}
    """
    print("\n" + "=" * 60)
    print("[阶段二] 分层秩分配 (Hierarchical Rank Allocation)...")
    if layer_sensitivity:
        print(f"第一级策略: Layer Sensitivity (Block Influence), Temp={temperature}")
    else:
        print("第一级策略: Spectral Entropy (Fallback)")
    print("第二级策略: Module Spectral Entropy")
    print("=" * 60)
    
    # 统计原始参数量
    total_original_params = 0
    layer_original_params = {}
    
    for key, info in svd_info.items():
        layer_idx, linear_name = key
        params = info.rows * info.cols
        total_original_params += params
        
        if layer_idx not in layer_original_params:
            layer_original_params[layer_idx] = 0
        layer_original_params[layer_idx] += params
    
    target_params = int(total_original_params * target_param_ratio)
    print(f"原始参数量: {total_original_params:,}")
    print(f"目标参数量: {target_params:,} (比例: {target_param_ratio:.2%})")
    
    # 第一级：计算每层的参数预算
    print("\n--- 第一级: 层级参数预算分配 ---")
    
    layer_importance = compute_layer_importance(svd_info, spectral_features, layer_sensitivity, temperature)
    
    # 计算总的加权分数 (基于乘法逻辑: 参数占比 * 重要性)
    # 这样 Importance 直接作为压缩率的缩放系数
    # total_weighted_score = sum(
    #     (layer_original_params[l] / total_original_params) * layer_importance[l]
    #     for l in layer_importance
    # )
    
    # 1. 提取所有层的重要性
    sorted_layers = sorted(layer_importance.keys())
    importances = np.array([layer_importance[l] for l in sorted_layers])
    
    # [Auto] 如果 layer_fluctuation_ratio < 0, 则自动计算
    if layer_fluctuation_ratio < 0:
        print("Auto-tuning layer_fluctuation_ratio...")
        layer_fluctuation_ratio = auto_determine_fluctuation_ratio(
            importances, target_param_ratio, min_rank_ratio, max_rank_ratio
        )
        print(f"Determined layer_fluctuation_ratio: {layer_fluctuation_ratio:.4f}")

    # 2. 归一化并去均值 (Min-Max Normalize -> Zero Center)
    if len(importances) > 1 and (importances.max() > importances.min()):
        # 归一化到 [0, 1]
        imp_norm = (importances - importances.min()) / (importances.max() - importances.min())
        # 缩放幅度 (Lambda), 控制重要性对最终压缩率的影响力
        # 如果 Lambda 很大，重要层和不重要层的压缩率差异会很大
        imp_lambda = layer_fluctuation_ratio 
        imp_scaled = imp_norm * imp_lambda
        # 去均值
        imp_centered = imp_scaled - np.mean(imp_scaled)
    else:
        imp_centered = np.zeros_like(importances)
        
    # 3. 计算每层的目标保留比例
    # base_ratio = target_param_ratio
    # layer_ratio = base_ratio + importance_shift
    # 重要性越高(>0), 保留比例越高; 重要性越低(<0), 保留比例越低
    target_ratios = target_param_ratio + imp_centered
    
    # 4. 全局参数量校准 (Global Calibration)
    # 由于各层参数量不同，简单的去均值不能保证总参数量守恒
    # 需要计算加权后的总参数量差异，并平摊回去
    layer_params_arr = np.array([layer_original_params[l] for l in sorted_layers])
    
    # 当前分配方案下的预计总参数量
    current_total = np.sum(target_ratios * layer_params_arr)
    target_total = target_param_ratio * np.sum(layer_params_arr)
    
    # 计算差异并修正
    diff_params = target_total - current_total
    # 将差异平均分摊到每一层 (或者按参数量权重分摊，这里选择平均分摊以保持简单)
    # 也可以选择 diff_ratio = diff_params / np.sum(layer_params_arr)
    diff_ratio = diff_params / np.sum(layer_params_arr)
    target_ratios += diff_ratio
    
    # 5. 截断 (Clipping)
    target_ratios = np.clip(target_ratios, min_rank_ratio, max_rank_ratio)
    
    # 构建最终的 budget 字典
    layer_param_budget = {}
    for i, layer_idx in enumerate(sorted_layers):
        ratio = target_ratios[i]
        layer_param_budget[layer_idx] = int(layer_original_params[layer_idx] * ratio)

    # 旧逻辑注释掉
    # layer_param_budget = {}
    # for layer_idx, importance in layer_importance.items():
    #     # 1. 基础比例 (如果不考虑重要性，应分配的比例)
    #     layer_ratio = layer_original_params[layer_idx] / total_original_params
    #     
    #     # 2. 结合重要性 (乘法缩放)
    #     weight = layer_ratio * importance
    #     
    #     # 3. 归一化分配预算
    #     layer_param_budget[layer_idx] = int(target_params * (weight / total_weighted_score))
    
    for layer_idx in sorted(layer_param_budget.keys())[:5]:
        orig = layer_original_params[layer_idx]
        budget = layer_param_budget[layer_idx]
        ratio = budget / orig if orig > 0 else 0
        imp = layer_importance[layer_idx]
        print(f"  Layer {layer_idx}: 原始={orig:,}, 预算={budget:,}, "
              f"压缩率={ratio:.2%}, 重要性={imp:.4f}")
    if len(layer_param_budget) > 5:
        print(f"  ... (共 {len(layer_param_budget)} 层)")
    
    # 第二级：层内模块秩分配
    print("\n--- 第二级: 层内模块秩分配 ---")
    
    # [Auto] 如果 module_fluctuation_ratio < 0, 则自动计算
    if module_fluctuation_ratio < 0:
        print("Auto-tuning module_fluctuation_ratio...")
        possible_ks = []
        
        for layer_idx in sorted(layer_param_budget.keys()):
            layer_budget = layer_param_budget[layer_idx]
            
            # Find modules in this layer
            layer_modules_keys = [
                key for key in svd_info.keys() if key[0] == layer_idx
            ]
            
            if not layer_modules_keys:
                continue
                
            # Compute module importance
            mod_imp_dict = compute_intra_layer_importance(svd_info, layer_idx, spectral_features)
            
            # Prepare data for auto function
            # Consistent sorting
            sorted_mod_keys = sorted(layer_modules_keys)
            m_imps = np.array([mod_imp_dict.get(k[1], 1.0/len(layer_modules_keys)) for k in sorted_mod_keys])
            
            # Calculate layer actual target ratio
            layer_orig_params = sum(svd_info[k].rows * svd_info[k].cols for k in layer_modules_keys)
            if layer_orig_params > 0:
                layer_actual_ratio = layer_budget / layer_orig_params
            else:
                layer_actual_ratio = 0
            
            # Calculate safe k for this layer
            k = auto_determine_fluctuation_ratio(
                m_imps, layer_actual_ratio, min_rank_ratio, max_rank_ratio
            )
            possible_ks.append(k)
            
        if possible_ks:
            module_fluctuation_ratio = min(possible_ks)
        else:
            module_fluctuation_ratio = 0.0
            
        print(f"Determined module_fluctuation_ratio: {module_fluctuation_ratio:.4f}")

    rank_allocation = {}
    
    for layer_idx in sorted(layer_param_budget.keys()):
        layer_budget = layer_param_budget[layer_idx]
        
        layer_modules = {
            key: info for key, info in svd_info.items() 
            if key[0] == layer_idx
        }
        
        if not layer_modules:
            continue
        
        module_importance = compute_intra_layer_importance(svd_info, layer_idx, spectral_features)
        layer_orig_params = sum(info.rows * info.cols for info in layer_modules.values())
        
        # 计算层内总加权分数 (用于归一化)
        # Score = (ModuleParams / LayerParams) * ModuleImportance
        # total_layer_score = sum(
        #     ((svd_info[k].rows * svd_info[k].cols) / layer_orig_params) * 
        #     module_importance.get(k[1], 1.0 / len(layer_modules))
        #     for k in layer_modules
        # )

        # 1. 准备数据
        sorted_modules = sorted(layer_modules.keys()) # 排序保证确定性
        m_importances = np.array([module_importance.get(k[1], 1.0/len(layer_modules)) for k in sorted_modules])
        m_params = np.array([svd_info[k].rows * svd_info[k].cols for k in sorted_modules])
        
        # 该层的目标压缩率 (Input Budget Ratio)
        layer_target_ratio = layer_budget / layer_orig_params if layer_orig_params > 0 else 0
        
        # 2. 归一化并去均值
        if len(m_importances) > 1 and (m_importances.max() > m_importances.min()):
            m_imp_norm = (m_importances - m_importances.min()) / (m_importances.max() - m_importances.min())
            # 层内调节幅度 Lambda
            m_imp_lambda = module_fluctuation_ratio
            m_imp_scaled = m_imp_norm * m_imp_lambda
            m_imp_centered = m_imp_scaled - np.mean(m_imp_scaled)
        else:
            m_imp_centered = np.zeros_like(m_importances)
            
        # 3. 计算模块目标比率
        # module_ratio = layer_ratio + shift
        m_target_ratios = layer_target_ratio + m_imp_centered
        
        # 4. 层内参数量校准
        current_m_total = np.sum(m_target_ratios * m_params)
        target_m_total = layer_budget # 必须严格等于分配给该层的 budget
        
        m_diff = target_m_total - current_m_total
        # 平均分摊差异 (或按参数量权重分摊)
        if np.sum(m_params) > 0:
            m_diff_ratio = m_diff / np.sum(m_params)
            m_target_ratios += m_diff_ratio
            
        # 5. 截断 (Clipping)
        # 这里的 min/max ratio 可以沿用全局配置，或者更宽松一些
        m_target_ratios = np.clip(m_target_ratios, min_rank_ratio, max_rank_ratio)
        
        # 分配 Rank
        for i, key in enumerate(sorted_modules):
            info = svd_info[key]
            target_ratio = m_target_ratios[i]
            
            # 计算该模块的 budget
            module_params = info.rows * info.cols
            module_budget = int(module_params * target_ratio)
            
            # 原有的逻辑
            # linear_name = key[1]
            # importance = module_importance.get(linear_name, 1.0 / len(layer_modules))
            
            # module_orig_params = info.rows * info.cols
            # module_ratio = module_orig_params / layer_orig_params if layer_orig_params > 0 else 1.0 / len(layer_modules)
            
            # 乘法逻辑分配: Budget = LayerBudget * (Score / TotalScore)
            # 其中 Score = Ratio * Importance
            # score = module_ratio * importance
            
            # module_budget = layer_budget * (score / total_layer_score) if total_layer_score > 0 else 0
            
            rank = int(module_budget / (info.rows + info.cols))
            
            max_rank = min(info.rows, info.cols)
            
            # 计算无膨胀的最大秩 (No-Inflation Rank Limit)
            # rank * (rows + cols) <= rows * cols
            # rank <= rows * cols / (rows + cols)
            max_no_inflation_rank = int(info.rows * info.cols / (info.rows + info.cols))
            
            min_rank = max(1, int(max_rank * min_rank_ratio))
            
            # 综合限制: 不超过 max_rank_ratio, 且不超过 no_inflation_rank
            max_allowed_rank = min(int(max_rank * max_rank_ratio), max_no_inflation_rank)
            
            rank = np.clip(rank, min_rank, max_allowed_rank)
            rank_allocation[key] = rank
    
    # 3. 迭代式余量填充 (Iterative Residual Filling)
    # 目的: 将因约束而未使用的预算重新分配给未饱和的模块，以逼近 target_params
    print("\n--- 阶段二(补充): 迭代式余量填充 ---")
    
    current_params = sum(
        rank_allocation[key] * (svd_info[key].rows + svd_info[key].cols)
        for key in rank_allocation
    )
    
    max_iterations = 5
    for i in range(max_iterations):
        gap = target_params - current_params
        if gap <= 0:
            break
            
        print(f"Iteration {i+1}: 剩余预算 {gap:,} (当前比例 {current_params/total_original_params:.2%})")
        
        # 找出所有未饱和的模块
        unsaturated_modules = []
        total_weight = 0
        
        for key, rank in rank_allocation.items():
            info = svd_info[key]
            max_rank = min(info.rows, info.cols)
            max_no_inflation_rank = int(info.rows * info.cols / (info.rows + info.cols))
            max_allowed_rank = min(int(max_rank * max_rank_ratio), max_no_inflation_rank)
            
            if rank < max_allowed_rank:
                # 权重结合参数量和重要性，确保余量优先分配给重要的模块
                # weight = params * layer_imp * module_imp
                l_idx = key[0]
                m_name = key[1]
                
                l_imp = layer_importance.get(l_idx, 1.0)
                
                # 需要重新获取 module_importance，这里稍微有些低效但逻辑正确
                # 为了简化，我们假设之前计算的 module_importance 已被缓存或重新快速计算
                # 这里暂时用一种近似方法：从 svd_info 或 spectral_features 获取 spectral_entropy
                m_imp = 1.0
                if spectral_features and key in spectral_features:
                    m_imp = spectral_features[key].spectral_entropy
                
                weight = (info.rows * info.cols) * l_imp * m_imp
                unsaturated_modules.append((key, weight, max_allowed_rank))
                total_weight += weight
        
        if not unsaturated_modules:
            print("所有模块已饱和，无法继续填充。")
            break
            
        # 分配 Gap
        added_params = 0
        for key, weight, max_allowed in unsaturated_modules:
            # 分配到的参数增量
            param_quota = int(gap * weight / total_weight)
            if param_quota <= 0:
                continue
                
            info = svd_info[key]
            cost_per_rank = info.rows + info.cols
            rank_increase = int(param_quota / cost_per_rank)
            
            if rank_increase > 0:
                current_rank = rank_allocation[key]
                new_rank = min(current_rank + rank_increase, max_allowed)
                real_increase = new_rank - current_rank
                
                if real_increase > 0:
                    rank_allocation[key] = new_rank
                    added_params += real_increase * cost_per_rank
        
        current_params += added_params
        if added_params == 0:
            print("本轮未成功分配任何参数，停止迭代。")
            break

    # 统计最终结果
    current_params = sum(
        rank_allocation[key] * (svd_info[key].rows + svd_info[key].cols)
        for key in rank_allocation
    )
    
    print(f"\n✓ 分层秩分配完成")
    print(f"最终参数量: {current_params:,} (实际比例: {current_params/total_original_params:.2%})")
    
    # 打印层级压缩率统计
    print("\n--- 层级压缩率统计 ---")
    layer_compression = {}
    for key, rank in rank_allocation.items():
        layer_idx = key[0]
        info = svd_info[key]
        
        if layer_idx not in layer_compression:
            layer_compression[layer_idx] = {'original': 0, 'compressed': 0}
        
        layer_compression[layer_idx]['original'] += info.rows * info.cols
        layer_compression[layer_idx]['compressed'] += rank * (info.rows + info.cols)
    
    compression_ratios = []
    print(f"{'Layer':<6} | {'Importance':<12} | {'Ratio':<8} | {'Orig Params':<15} | {'Comp Params':<15}")
    print("-" * 70)
    
    # 获取层级重要性用于打印
    layer_importance = compute_layer_importance(svd_info, spectral_features, layer_sensitivity, temperature)
    
    for layer_idx in sorted(layer_compression.keys()):
        stats = layer_compression[layer_idx]
        ratio = stats['compressed'] / stats['original'] if stats['original'] > 0 else 0
        compression_ratios.append(ratio)
        importance = layer_importance.get(layer_idx, 0.0)
        print(f"{layer_idx:<6} | {importance:<12.4f} | {ratio:<8.2%} | {stats['original']:<15,} | {stats['compressed']:<15,}")
    
    print("-" * 70)
    print(f"层压缩率: min={min(compression_ratios):.2%}, max={max(compression_ratios):.2%}, "
          f"mean={np.mean(compression_ratios):.2%}, std={np.std(compression_ratios):.4f}")
    
    print_rank_statistics(svd_info, rank_allocation)
    
    # 打印每层内部的模块分配比例
    print("\n--- 层内模块分配详情 (Sample Layers) ---")
    
    # 打印所有层的详情
    sorted_layers = sorted(list(layer_compression.keys()))
    
    for l_idx in sorted_layers:
        print(f"\nLayer {l_idx} Details:")
        
        # 获取该层所有模块的秩分配
        layer_modules = {k: v for k, v in rank_allocation.items() if k[0] == l_idx}
        if not layer_modules:
            continue
            
        total_layer_params = layer_compression[l_idx]['compressed']
        
        print(f"{'Module':<20} | {'Rank':<6} | {'MaxRank':<8} | {'RankRatio':<10} | {'ParamShare':<10}")
        print("-" * 70)
        
        for k, rank in layer_modules.items():
            name = k[1]
            info = svd_info[k]
            max_r = min(info.rows, info.cols)
            rank_ratio = rank / max_r
            
            # 当前模块参数量 = rank * (rows + cols)
            current_module_params = rank * (info.rows + info.cols)
            param_share = current_module_params / total_layer_params if total_layer_params > 0 else 0
            
            print(f"{name:<20} | {rank:<6} | {max_r:<8} | {rank_ratio:<10.2%} | {param_share:<10.2%}")

    print("-" * 70)
    print(f"原始总参数量: {total_original_params:,}")
    print(f"压缩后总参数量: {current_params:,}")
    print(f"最终实际压缩比: {current_params/total_original_params:.2%}")
    print("-" * 70)

    return rank_allocation


# ============================================================================
# 阶段三：SVD 截断
# ============================================================================

def simple_svd_truncation(
    model,
    svd_info: Dict[Tuple[int, str], LayerSVDInfo],
    rank_allocation: Dict[Tuple[int, str], int],
    dev: str,
    enable_compensation: bool = False,
    compensation_limit: float = 2.0
) -> Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    根据秩分配进行 SVD 截断
    
    Args:
        ...
        enable_compensation: 是否启用能量补偿 (Energy Calibration)
        compensation_limit: 能量补偿的最大放大倍数限制 (Default: 2.0)
    """
    print("\n" + "=" * 60)
    print(f"[阶段三] SVD 截断 (Compensation: {enable_compensation}, Limit: {compensation_limit})...")
    print("=" * 60)
    
    compressed_weights = {}
    dtype = next(iter(model.parameters())).dtype
    
    # 统计 kept_energy
    kept_energy_stats = []
    
    for key, rank in tqdm(rank_allocation.items(), desc="Truncation"):
        info = svd_info[key]
        
        trunc_U = info.U[:, :rank].float().to(dev)
        trunc_S = info.S[:rank].float().to(dev)
        
        # 能量补偿逻辑
        if enable_compensation:
            original_energy = torch.sum(info.S.to(dev) ** 2)
            kept_energy = torch.sum(trunc_S ** 2)
            
            kept_energy_val = kept_energy.item()
            kept_energy_stats.append(kept_energy_val)
            
            if kept_energy > 1e-6:
                scale_factor = torch.sqrt(original_energy / kept_energy)
                # 限制放大倍数，防止过度放大噪声 (例如最大放大 2 倍)
                scale_factor = torch.clamp(scale_factor, 1.0, compensation_limit)
                trunc_S = trunc_S * scale_factor
        
        trunc_VT = info.VT[:rank, :].float().to(dev)
        scaling_matrix_inv = info.scaling_matrix_inv.float().to(dev)
        
        trunc_V = torch.matmul(trunc_VT, scaling_matrix_inv)
        
        truc_sigma = torch.diag(trunc_S)
        sqrtSigma = torch.sqrt(truc_sigma)
        
        final_U = torch.matmul(trunc_U, sqrtSigma)
        final_V = torch.matmul(sqrtSigma, trunc_V)
        
        compressed_weights[key] = (final_U.cpu().to(dtype), final_V.cpu().to(dtype))
        
        del trunc_U, trunc_S, trunc_VT, trunc_V, truc_sigma, sqrtSigma, final_U, final_V, scaling_matrix_inv
        torch.cuda.empty_cache()
        
    if enable_compensation and kept_energy_stats:
        total_count = len(kept_energy_stats)
        gt_1e6_count = sum(1 for x in kept_energy_stats if x > 1e-6)
        print("\n" + "=" * 40)
        print("Energy Compensation Statistics:")
        print(f"Total layers processed: {total_count}")
        print(f"kept_energy > 1e-6: {gt_1e6_count} ({gt_1e6_count/total_count:.2%})")
        print(f"Min kept_energy: {min(kept_energy_stats):.6e}")
        print(f"Max kept_energy: {max(kept_energy_stats):.6e}")
        print(f"Mean kept_energy: {sum(kept_energy_stats)/total_count:.6e}")
        print("=" * 40 + "\n")

    print("✓ SVD 截断完成")
    return compressed_weights


# ============================================================================
# 阶段四：低秩算子替换
# ============================================================================

@torch.no_grad()
def replace_with_low_rank_layers(
    model_name: str,
    model,
    compressed_weights: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]],
    rank_allocation: Dict[Tuple[int, str], int],
    ratio: float
):
    """
    将压缩后的权重替换到模型中，使用低秩层结构
    """
    print("\n" + "=" * 60)
    print("[阶段四] 低秩算子替换...")
    print("=" * 60)
    
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    
    for i in tqdm(range(len(layers)), desc="Replace"):
        layer = layers[i]
        subset = find_layers(layer)
        
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(
                hidden_size=model.config.hidden_size,
                intermediate_size=model.config.intermediate_size,
                hidden_act=model.config.hidden_act,
                ratio=ratio
            )
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        
        for name in subset:
            key = (i, name)
            if key not in compressed_weights:
                continue
            
            svd_u, svd_v = compressed_weights[key]
            
            if 'opt' in model_name:
                if "q_proj" in name:
                    _resize_and_set_weights(svd_decoder.self_attn.q_u_proj, svd_decoder.self_attn.q_v_proj, svd_u, svd_v)
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data
                elif "k_proj" in name:
                    _resize_and_set_weights(svd_decoder.self_attn.k_u_proj, svd_decoder.self_attn.k_v_proj, svd_u, svd_v)
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    _resize_and_set_weights(svd_decoder.self_attn.v_u_proj, svd_decoder.self_attn.v_v_proj, svd_u, svd_v)
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    _resize_and_set_weights(svd_decoder.self_attn.out_u_proj, svd_decoder.self_attn.out_v_proj, svd_u, svd_v)
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    _resize_and_set_weights(svd_decoder.fc1_u_proj, svd_decoder.fc1_v_proj, svd_u, svd_v)
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    _resize_and_set_weights(svd_decoder.fc2_u_proj, svd_decoder.fc2_v_proj, svd_u, svd_v)
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    _resize_and_set_weights(svd_attn.q_u_proj, svd_attn.q_v_proj, svd_u, svd_v)
                elif "k_proj" in name:
                    _resize_and_set_weights(svd_attn.k_u_proj, svd_attn.k_v_proj, svd_u, svd_v)
                elif "v_proj" in name:
                    _resize_and_set_weights(svd_attn.v_u_proj, svd_attn.v_v_proj, svd_u, svd_v)
                elif "o_proj" in name:
                    _resize_and_set_weights(svd_attn.o_u_proj, svd_attn.o_v_proj, svd_u, svd_v)
                    layer.self_attn = svd_attn
                elif "gate_proj" in name:
                    _resize_and_set_weights(svd_mlp.gate_u_proj, svd_mlp.gate_v_proj, svd_u, svd_v)
                elif "down_proj" in name:
                    _resize_and_set_weights(svd_mlp.down_u_proj, svd_mlp.down_v_proj, svd_u, svd_v)
                elif "up_proj" in name:
                    _resize_and_set_weights(svd_mlp.up_u_proj, svd_mlp.up_v_proj, svd_u, svd_v)
                    layer.mlp = svd_mlp
        
        torch.cuda.empty_cache()
    
    print("✓ 低秩层替换完成")


def _resize_and_set_weights(u_layer: nn.Linear, v_layer: nn.Linear, 
                            svd_u: torch.Tensor, svd_v: torch.Tensor):
    """动态调整层大小并设置权重"""
    u_layer.weight.data = svd_u
    v_layer.weight.data = svd_v


# ============================================================================
# 辅助函数：PPL 自动搜索
# ============================================================================

@torch.no_grad()
def update_model_weights_from_svd(
    model,
    svd_info: Dict[Tuple[int, str], LayerSVDInfo],
    rank_allocation: Dict[Tuple[int, str], int],
    dev: str
):
    """
    根据分配的秩，重构权重并更新到模型中 (用于 PPL 评估)
    """
    if hasattr(model, 'model'):
        if hasattr(model.model, 'decoder'): # OPT
             layers = model.model.decoder.layers
        else: # Llama, Mistral
             layers = model.model.layers
    else:
        # Fallback
        layers = model.model.layers

    for key, rank in rank_allocation.items():
        layer_idx, linear_name = key
        info = svd_info[key]
        
        # SVD 重构: W ~ U_r * S_r * VT_r * ScalingInv
        # 注意: info.U 等在 CPU
        
        U_r = info.U[:, :rank].to(dev)
        S_r = info.S[:rank].to(dev)
        VT_r = info.VT[:rank, :].to(dev)
        ScalingInv = info.scaling_matrix_inv.to(dev)
        
        # W_recon = (U * S * VT) * ScalingInv
        W_recon = torch.matmul(U_r, torch.diag(S_r))
        W_recon = torch.matmul(W_recon, VT_r)
        W_recon = torch.matmul(W_recon, ScalingInv)
        
        # 更新模型权重
        layer = layers[layer_idx]
        subset = find_layers(layer)
        if linear_name in subset:
            # 确保将权重转换回该层参数所在的设备
            target_device = subset[linear_name].weight.device
            subset[linear_name].weight.data = W_recon.to(target_device).to(subset[linear_name].weight.dtype)
        
        del U_r, S_r, VT_r, ScalingInv, W_recon
    
    torch.cuda.empty_cache()


@torch.no_grad()
def compute_calibration_ppl(
    model, 
    tokenizer, 
    calib_loader, 
    dev: str,
    n_samples: int = 16
) -> float:
    """
    计算校准数据集上的 PPL
    """
    model.eval()
    nalls = 0
    nloss = 0
    
    # 获取模型所在的设备
    try:
        model_device = next(model.parameters()).device
    except:
        model_device = torch.device('cpu')

    # 只取前 n_samples 个样本
    count = 0
    
    # calib_loader 是一个 list of tensors
    for batch in calib_loader:
        if count >= n_samples:
            break
            
        if torch.is_tensor(batch):
             inp = batch.to(model_device)
        elif isinstance(batch, dict):
             inp = batch['input_ids'].to(model_device)
        else:
             continue
             
        if inp.dim() == 2:
            pass
        elif inp.dim() == 3:
            inp = inp.squeeze(0)
            
        lm_logits = model(inp).logits
        
        # Shift logits and labels
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inp[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        nloss += loss.item() * inp.size(1) # Loss * SeqLen
        nalls += inp.size(1)
        
        count += 1
        
    return np.exp(nloss / nalls)


def auto_determine_fluctuation_ratio_ppl(
    model,
    model_name: str,
    tokenizer,
    svd_info: Dict,
    spectral_features: Dict,
    target_param_ratio: float,
    calib_loader,
    dev: str,
    min_rank_ratio: float = 0.05,
    max_rank_ratio: float = 0.95,
    layer_sensitivity: Dict = None,
    temperature: float = 0.5,
    search_l_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    search_m_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    n_calib_samples: int = 8
) -> Tuple[float, float]:
    """
    使用 Grid Search + PPL 自动确定最佳的 fluctuation ratios
    """
    print("\n" + "="*60)
    print(f"[Auto-Tuning] 基于 PPL 的参数搜索 (Samples={n_calib_samples})...")
    print("="*60)
    
    # # 记录原始设备并在搜索前将模型移动到计算设备 (加速推理)
    # try:
    #     original_device = next(model.parameters()).device
    # except:
    #     original_device = torch.device('cpu')
    
    # print(f"Moving model to {dev} for fast PPL evaluation...")
    # try:
    #     model.to(dev)
    # except Exception as e:
    #     print(f"Warning: Failed to move model to {dev}: {e}")
    #     print("Continuing on original device (might be slow)...")
    model.to(dev)
    best_ppl = float('inf')
    # Initialize with median values or defaults
    best_l_ratio = search_l_ratios[len(search_l_ratios) // 2] if search_l_ratios else 0.5
    best_m_ratio = search_m_ratios[len(search_m_ratios) // 2] if search_m_ratios else 0.5
    
    total_steps = len(search_l_ratios) + len(search_m_ratios)
    step = 0
    
    def evaluate_single_config(l_r, m_r, current_step, total_steps_count):
        nonlocal best_ppl, best_l_ratio, best_m_ratio
        
        print(f"[{current_step}/{total_steps_count}] Testing L_Ratio={l_r}, M_Ratio={m_r}...", end="")
        sys.stdout.flush()
        
        # 1. 计算秩分配 (Suppress Output)
        with contextlib.redirect_stdout(io.StringIO()):
            rank_allocation = hierarchical_rank_allocation(
                svd_info,
                target_param_ratio=target_param_ratio,
                min_rank_ratio=min_rank_ratio,
                max_rank_ratio=max_rank_ratio,
                spectral_features=spectral_features,
                layer_sensitivity=layer_sensitivity,
                temperature=temperature,
                layer_fluctuation_ratio=l_r,
                module_fluctuation_ratio=m_r
            )
        
        # 2. 更新权重
        update_model_weights_from_svd(model, svd_info, rank_allocation, dev)
        
        # 3. 计算 PPL
        try:
            ppl = compute_calibration_ppl(model, tokenizer, calib_loader, dev, n_samples=n_calib_samples)
        except Exception as e:
            print(f" Error: {e}")
            ppl = float('inf')
        
        print(f" PPL={ppl:.4f}")
        
        if ppl < best_ppl:
            best_ppl = ppl
            best_l_ratio = l_r
            best_m_ratio = m_r
        
        return ppl

    try:
        # Phase 1: Coordinate Descent - Search L (Fix M)
        print(f"Phase 1: Searching Layer Ratio (Fixed Module Ratio = {best_m_ratio})...")
        current_fixed_m = best_m_ratio
        for l_ratio in search_l_ratios:
            step += 1
            ppl = evaluate_single_config(l_ratio, current_fixed_m, step, total_steps)
            
            # 拐点检测：如果 PPL 变大，说明过了最佳点，提前结束
            if ppl > best_ppl:
                print(f"Inflection point detected (PPL increased from {best_ppl:.4f} to {ppl:.4f}). Stopping Phase 1.")
                break
            
        # Phase 2: Coordinate Descent - Search M (Fix L)
        print(f"Phase 2: Searching Module Ratio (Fixed Layer Ratio = {best_l_ratio})...")
        current_fixed_l = best_l_ratio
        
        # Reset inflection point detection for Phase 2
        # 我们使用 phase2_best_ppl 来判断是否在 Phase 2 的搜索路径上出现了 PPL 上升
        # 即使当前的 PPL 可能比全局 best_ppl 差，我们也不应该立即停止，而是要看 Phase 2 内部的趋势
        phase2_best_ppl = float('inf')
        
        for m_ratio in search_m_ratios:
            step += 1
            ppl = evaluate_single_config(current_fixed_l, m_ratio, step, total_steps)
            
            # 更新本阶段最佳
            if ppl < phase2_best_ppl:
                phase2_best_ppl = ppl
            
            # 拐点检测：如果 PPL 比本阶段见过的最佳值要大，说明过了拐点
            if ppl > phase2_best_ppl:
                print(f"Inflection point detected (PPL increased from {phase2_best_ppl:.4f} to {ppl:.4f}). Stopping Phase 2.")
                break
    
    finally:
        # print(f"Restoring model device to {original_device}...")
        # model.to(original_device)
        torch.cuda.empty_cache()
    
    print("-" * 60)
    print(f"Best PPL: {best_ppl:.4f} @ L_Ratio={best_l_ratio}, M_Ratio={best_m_ratio}")
    print("=" * 60 + "\n")
    
    return best_l_ratio, best_m_ratio


# ============================================================================
# 主函数：基于 spectral_entropy 的 SVD 压缩
# ============================================================================

@torch.no_grad()
def cgsvr_compress(
    model_name: str,
    model,
    tokenizer,
    target_ratio: float,
    calib_dataset: str = 'wikitext2',
    whitening_nsamples: int = 256,
    seqlen: int = 2048,
    seed: int = 0,
    dev: str = 'cuda',
    save_path: str = None,
    profiling_mat_path: str = None,
    min_rank_ratio: float = 0.05,
    max_rank_ratio: float = 0.95,
    importance_metric: str = "cosine",
    enable_compensation: bool = False,
    compensation_limit: float = 2.0,
    temperature: float = 0.5,
    layer_fluctuation_ratio: float = 0.5,
    module_fluctuation_ratio: float = 0.5,
    use_ppl_auto_ratio: bool = False
):
    """
    基于 spectral_entropy 的 SVD 压缩流程
    
    Args:
        ...
        importance_metric: 层级重要性度量标准 ['cosine', 'angular']
        enable_compensation: 是否开启 SVD 截断后的能量补偿
        compensation_limit: 能量补偿上限
        temperature: 平滑温度系数
        layer_fluctuation_ratio: 层间波动系数
        module_fluctuation_ratio: 层内波动系数
        use_ppl_auto_ratio: 是否使用 PPL 自动搜索波动系数
    """
    print("\n" + "=" * 70)
    print("Spectral Entropy SVD Compression")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"目标压缩比例: {target_ratio:.2%}")
    print(f"重要性度量: {importance_metric}")
    print(f"能量补偿: {enable_compensation} (Limit: {compensation_limit})")
    print(f"温度系数: {temperature}")
    print(f"层间波动系数: {layer_fluctuation_ratio}")
    print(f"层内波动系数: {module_fluctuation_ratio}")
    print(f"PPL自动搜索: {use_ppl_auto_ratio}")
    print("=" * 70 + "\n")
    
    model.eval()
    
    # ====== 第一阶段：数据感知白化 ======
    default_profiling_path = None
    default_sensitivity_path = None
    if save_path is not None:
        base_name = f"{model_name.replace('/', '_').replace('-', '_')}"
        default_profiling_path = os.path.join(save_path, 
            f"{base_name}_cgsvr_profiling_{calib_dataset}_{whitening_nsamples}.pt")
        print(default_profiling_path)
        # 不同的 metric 使用不同的文件名保存
        default_sensitivity_path = os.path.join(save_path, 
            f"{base_name}_cgsvr_sensitivity_{importance_metric}_{calib_dataset}_{whitening_nsamples}.pt")
    
    load_path = None
    if profiling_mat_path is not None and os.path.exists(profiling_mat_path):
        load_path = profiling_mat_path
    elif default_profiling_path is not None and os.path.exists(default_profiling_path):
        load_path = default_profiling_path
    
    if load_path is not None:
        print(f"加载预计算的白化矩阵: {load_path}")
        # 简化逻辑: 只加载纯白化矩阵，不再尝试从旧文件恢复 similarity
        loaded_data = torch.load(load_path)
        if isinstance(loaded_data, dict) and 'profiling_mat' in loaded_data:
            profiling_mat = loaded_data['profiling_mat']
        else:
            profiling_mat = loaded_data
            
        layer_sensitivity = {}

        # 尝试加载单独的 sensitivity 文件
        if default_sensitivity_path is not None and os.path.exists(default_sensitivity_path):
            print(f"加载预计算的层级灵敏度: {default_sensitivity_path}")
            layer_sensitivity = torch.load(default_sensitivity_path)
            
        # 如果依然没有灵敏度数据，则计算并保存
        if not layer_sensitivity:
            print("计算层级灵敏度...")
            calib_data = get_calib_train_data(calib_dataset, tokenizer, whitening_nsamples, seqlen=seqlen, seed=seed)
            layer_sensitivity = compute_layer_sensitivity(model_name, model, calib_data, dev, metric=importance_metric)
            
            if default_sensitivity_path:
                print(f"保存层级灵敏度: {default_sensitivity_path}")
                torch.save(layer_sensitivity, default_sensitivity_path)
    else:
        calib_data = get_calib_train_data(calib_dataset, tokenizer, whitening_nsamples, seqlen=seqlen, seed=seed)
        profiling_mat = compute_whitening_matrices(model_name, model, calib_data, dev)
        layer_sensitivity = compute_layer_sensitivity(model_name, model, calib_data, dev, metric=importance_metric)
        
        if default_profiling_path is not None:
            os.makedirs(os.path.dirname(default_profiling_path), exist_ok=True)
            torch.save(profiling_mat, default_profiling_path)
            print(f"白化矩阵已保存: {default_profiling_path}")
            
        if default_sensitivity_path is not None:
            os.makedirs(os.path.dirname(default_sensitivity_path), exist_ok=True)
            torch.save(layer_sensitivity, default_sensitivity_path)
            print(f"层级灵敏度已保存: {default_sensitivity_path}")
    
    # ====== 第二阶段：SVD 分解 + 秩分配 ======
    svd_info = compute_svd_and_importance(model_name, model, profiling_mat, dev, save_path=save_path)
    
    # 基于 spectral_entropy 的分层秩分配
    spectral_features = extract_spectral_features(svd_info)
    
    if use_ppl_auto_ratio:
        print("\n启用 PPL 自动搜索波动系数...")
        if 'calib_data' not in locals():
             print("Loading calibration data for PPL search...")
             calib_data = get_calib_train_data(calib_dataset, tokenizer, whitening_nsamples, seqlen=seqlen, seed=seed)
             
        layer_fluctuation_ratio, module_fluctuation_ratio = auto_determine_fluctuation_ratio_ppl(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            svd_info=svd_info,
            spectral_features=spectral_features,
            target_param_ratio=target_ratio,
            calib_loader=calib_data,
            dev=dev,
            min_rank_ratio=min_rank_ratio,
            max_rank_ratio=max_rank_ratio,
            layer_sensitivity=layer_sensitivity,
            temperature=temperature
        )
        print(f"PPL 搜索完成: layer_ratio={layer_fluctuation_ratio}, module_ratio={module_fluctuation_ratio}")

    rank_allocation = hierarchical_rank_allocation(
        svd_info,
        target_param_ratio=target_ratio,
        min_rank_ratio=min_rank_ratio,
        max_rank_ratio=max_rank_ratio,
        spectral_features=spectral_features,
        layer_sensitivity=layer_sensitivity,
        temperature=temperature,
        layer_fluctuation_ratio=layer_fluctuation_ratio,
        module_fluctuation_ratio=module_fluctuation_ratio
    )
    
    # ====== 阶段三：SVD 截断 ======
    compressed_weights = simple_svd_truncation(
        model, 
        svd_info, 
        rank_allocation, 
        dev,
        enable_compensation=enable_compensation,
        compensation_limit=compensation_limit
    )
    
    # ====== 阶段四：低秩层替换 ======
    replace_with_low_rank_layers(model_name, model, compressed_weights, rank_allocation, target_ratio)
    
    # 保存模型
    # if save_path is not None:
    #     os.makedirs(save_path, exist_ok=True)
    #     save_file = os.path.join(save_path, 
    #         f"{model_name.replace('/', '_').replace('-', '_')}_effrank_{target_ratio}.pt")
    #     save_dict = {
    #         'model': model, 
    #         'tokenizer': tokenizer, 
    #         'rank_allocation': rank_allocation
    #     }
    #     torch.save(save_dict, save_file)
    #     print(f"\n✓ 压缩模型已保存: {save_file}")
    
    print("\n" + "=" * 70)
    print("Spectral Entropy SVD 压缩完成!")
    print("=" * 70)
    
    return model, rank_allocation


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectral Entropy SVD 压缩')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True, help='模型名称或 HuggingFace 路径')
    parser.add_argument('--model_path', type=str, default=None, help='本地压缩模型路径')
    
    # 压缩参数
    parser.add_argument('--ratio', type=float, default=0.2, 
                       help='压缩比例 (0,1)，如 0.2 表示压缩掉 20%% 参数（保留 80%%）')
    parser.add_argument('--min_rank_ratio', type=float, default=0.05, 
                       help='每个矩阵的最小秩比例')
    parser.add_argument('--max_rank_ratio', type=float, default=0.95, 
                       help='每个矩阵的最大秩比例')
    parser.add_argument('--importance_metric', type=str, default='cosine', choices=['cosine', 'angular'],
                       help='层级重要性度量标准: cosine (1-sim, all tokens) 或 angular (arccos, last token)')
    parser.add_argument('--enable_compensation', action='store_true', help='是否开启 SVD 截断后的能量补偿')
    parser.add_argument('--compensation_limit', type=float, default=2.0, help='能量补偿的最大放大倍数限制')
    parser.add_argument('--temperature', type=float, default=0.5, help='重要性平滑温度系数')
    parser.add_argument('--layer_fluctuation_ratio', type=float, default=0.5, help='层间重要性波动系数')
    parser.add_argument('--module_fluctuation_ratio', type=float, default=0.5, help='层内重要性波动系数')
    parser.add_argument('--use_ppl', action='store_true', help='使用 PPL 自动确定波动系数')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='wikitext2', 
                       help='校准数据集 [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, 
                       help='白化校准样本数')
    parser.add_argument('--model_seq_len', type=int, default=2048, 
                       help='序列长度')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    
    # 保存/加载参数
    parser.add_argument('--save_path', type=str, default=None, help='保存路径')
    parser.add_argument('--profiling_mat_path', type=str, default=None, 
                       help='预计算白化矩阵路径')
    
    # 计算参数
    parser.add_argument('--DEV', type=str, default='cuda', help='计算设备')
    
    # 评估参数
    parser.add_argument('--step', type=int, default=0, 
                       help='运行步骤: 0=完整压缩, 4=评估PPL')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='评估批次大小')
    parser.add_argument('--tasks', type=str, default="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",help='Comma-separated list of evaluation tasks')
    args = parser.parse_args()
    
    # ratio 转换: 用户输入 0.2 表示压缩掉 20%，实际保留 80%
    args.ratio = 1 - args.ratio
    
    if args.step == 0:
        # 完整压缩流程
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.float()
        
        model, rank_allocation = cgsvr_compress(
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            target_ratio=args.ratio,
            calib_dataset=args.dataset,
            whitening_nsamples=args.whitening_nsamples,
            seqlen=args.model_seq_len,
            seed=args.seed,
            dev=args.DEV,
            save_path=args.save_path,
            profiling_mat_path=args.profiling_mat_path,
            min_rank_ratio=args.min_rank_ratio,
            max_rank_ratio=args.max_rank_ratio,
            importance_metric=args.importance_metric,
            enable_compensation=args.enable_compensation,
            compensation_limit=args.compensation_limit,
            temperature=args.temperature,
            layer_fluctuation_ratio=args.layer_fluctuation_ratio,
            module_fluctuation_ratio=args.module_fluctuation_ratio,
            use_ppl_auto_ratio=args.use_ppl
        )
        ppl_eval(model, tokenizer, datasets=['wikitext2', 'c4'],
                model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        zeroshot_eval(model, tokenizer, tasks=args.tasks,  batch_size=args.eval_batch_size, device=args.DEV)
        
    elif args.step >= 4:
        # 评估模式
        print(f"评估模型: {args.model_path}")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            checkpoint = torch.load(args.model_path, weights_only=False, map_location='cpu')
            model, tokenizer = checkpoint['model'], checkpoint['tokenizer']
            if 'rank_allocation' in checkpoint:
                print(f"秩分配信息: {len(checkpoint['rank_allocation'])} 个矩阵")
        
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        
        ppl_eval(model, tokenizer, datasets=['wikitext2','ptb','c4'],
                model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        zeroshot_eval(model, tokenizer, tasks=args.tasks,  batch_size=args.eval_batch_size, device=args.DEV)
