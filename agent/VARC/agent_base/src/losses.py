from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from .action_selector import CLICK_ACTION_ID, apply_joint_mask, expand_avail_to_joint
IGNORE_INDEX_AA3 = 16

def build_joint_logits(type_logits: torch.Tensor, click_logits: torch.Tensor) -> torch.Tensor:
    B = type_logits.size(0)
    return torch.cat([type_logits, click_logits.reshape(B, -1)], dim=-1)

def action_to_joint_index(action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, num_actions: int, H: int, W: int, click_id: int=CLICK_ACTION_ID) -> torch.Tensor:
    num_non_click = num_actions - 1
    non_click_idx = torch.where(action_id < click_id, action_id, action_id - 1)
    safe_x = action_x.clamp(min=0)
    safe_y = action_y.clamp(min=0)
    click_idx = num_non_click + safe_y * W + safe_x
    is_click = action_id == click_id
    return torch.where(is_click, click_idx, non_click_idx).long()

def joint_index_to_action(joint_idx: torch.Tensor, num_actions: int, H: int, W: int, click_id: int=CLICK_ACTION_ID) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_non_click = num_actions - 1
    is_click = joint_idx >= num_non_click
    non_click_action_id = torch.where(joint_idx < click_id, joint_idx, joint_idx + 1)
    click_offset = joint_idx - num_non_click
    y = click_offset // W
    x = click_offset % W
    action_id = torch.where(is_click, torch.full_like(joint_idx, click_id), non_click_action_id)
    x = torch.where(is_click, x, torch.full_like(x, -1))
    y = torch.where(is_click, y, torch.full_like(y, -1))
    return (action_id.long(), x.long(), y.long())

def ffp_loss(ffp_logits: torch.Tensor, targets: torch.Tensor, ignore_index: int=IGNORE_INDEX_AA3) -> torch.Tensor:
    num_colors = ffp_logits.size(1)
    logits_flat = ffp_logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
    return F.cross_entropy(logits_flat, targets.reshape(-1), ignore_index=ignore_index)

def bc_joint_loss(type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, *, num_actions: int=8, click_id: int=CLICK_ACTION_ID) -> Tuple[torch.Tensor, Dict[str, float]]:
    H, W = (click_logits.shape[-2], click_logits.shape[-1])
    joint_logits = build_joint_logits(type_logits, click_logits)
    joint_target = action_to_joint_index(action_id, action_x, action_y, num_actions, H, W, click_id=click_id)
    loss = F.cross_entropy(joint_logits, joint_target)
    with torch.no_grad():
        per_sample = F.cross_entropy(joint_logits, joint_target, reduction='none')
        is_click = action_id == click_id
        n_click = int(is_click.sum().item())
        loss_type_part = float(per_sample[~is_click].mean().item()) if (~is_click).any() else 0.0
        loss_click_part = float(per_sample[is_click].mean().item()) if n_click > 0 else 0.0
    parts = {'loss_bc': float(loss.detach()), 'loss_type': loss_type_part, 'loss_click': loss_click_part, 'n_click': float(n_click)}
    return (loss, parts)

def bc_loss(type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, avail_mask: Optional[torch.Tensor]=None, *, alpha_click: float=1.0, num_actions: int=8) -> Tuple[torch.Tensor, Dict[str, float]]:
    return bc_joint_loss(type_logits, click_logits, action_id, action_x, action_y, num_actions=num_actions)

def joint_loss(ffp_logits: torch.Tensor, targets: torch.Tensor, type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, avail_mask: Optional[torch.Tensor]=None, *, lambda_bc: float=1.0, alpha_click: float=1.0, num_actions: int=8, ignore_index: int=IGNORE_INDEX_AA3) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_ffp = ffp_loss(ffp_logits, targets, ignore_index=ignore_index)
    l_bc, bc_parts = bc_joint_loss(type_logits, click_logits, action_id, action_x, action_y, num_actions=num_actions)
    loss = l_ffp + lambda_bc * l_bc
    parts = {'loss_ffp': float(l_ffp.detach()), 'loss_bc': bc_parts['loss_bc'], 'loss_type': bc_parts['loss_type'], 'loss_click': bc_parts['loss_click'], 'loss_total': float(loss.detach())}
    return (loss, parts)

@torch.no_grad()
def ffp_metrics(ffp_logits: torch.Tensor, targets: torch.Tensor, ignore_index: int=IGNORE_INDEX_AA3) -> Dict[str, float]:
    pred = ffp_logits.argmax(dim=1)
    valid = targets != ignore_index
    correct = ((pred == targets) & valid).sum().item()
    total = valid.sum().item()
    acc_pixel = correct / max(total, 1)
    B = targets.size(0)
    exact = 0
    for b in range(B):
        v = valid[b]
        if v.any():
            exact += int(torch.equal(pred[b][v], targets[b][v]))
    acc_exact = exact / max(B, 1)
    return {'acc_pixel': acc_pixel, 'acc_exact': acc_exact}

@torch.no_grad()
def bc_metrics(type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, avail_mask: Optional[torch.Tensor]=None, *, click_radii: Tuple[int, ...]=(1, 3, 5), num_actions: int=8, click_id: int=CLICK_ACTION_ID) -> Dict[str, float]:
    B, H, W = click_logits.shape
    joint_logits = build_joint_logits(type_logits, click_logits)
    joint_target = action_to_joint_index(action_id, action_x, action_y, num_actions, H, W, click_id=click_id)
    if avail_mask is not None:
        joint_avail = expand_avail_to_joint(avail_mask, H, W, click_id=click_id)
        joint_logits_for_argmax = apply_joint_mask(joint_logits, joint_avail)
    else:
        joint_logits_for_argmax = joint_logits
    metrics: Dict[str, float] = {}
    pred_idx = joint_logits_for_argmax.argmax(dim=-1)
    pred_action_id, pred_x, pred_y = joint_index_to_action(pred_idx, num_actions, H, W, click_id=click_id)
    metrics['acc_type'] = (pred_action_id == action_id).float().mean().item()
    metrics['acc_joint'] = (pred_idx == joint_target).float().mean().item()
    logp = F.log_softmax(joint_logits_for_argmax, dim=-1)
    p = logp.exp()
    if avail_mask is not None:
        joint_avail_bool = joint_avail.bool()
        ent_terms = -(p * logp).masked_fill(~joint_avail_bool, 0.0)
    else:
        ent_terms = -(p * logp)
    metrics['entropy_type'] = ent_terms.sum(dim=-1).mean().item()
    is_click_true = action_id == click_id
    n_click = int(is_click_true.sum().item())
    if n_click > 0:
        true_x = action_x.clamp(min=0)[is_click_true].float()
        true_y = action_y.clamp(min=0)[is_click_true].float()
        pred_x_clk = pred_x[is_click_true].float()
        pred_y_clk = pred_y[is_click_true].float()
        pred_is_click = pred_action_id[is_click_true] == click_id
        exact = ((pred_x_clk == true_x) & (pred_y_clk == true_y) & pred_is_click).float().mean().item()
        metrics['acc_click'] = exact
        if pred_is_click.any():
            px = pred_x_clk[pred_is_click]
            py = pred_y_clk[pred_is_click]
            tx = true_x[pred_is_click]
            ty = true_y[pred_is_click]
            l2 = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)
            metrics['click_l2'] = l2.mean().item()
            linf = torch.maximum((px - tx).abs(), (py - ty).abs())
            for r in click_radii:
                metrics[f'click_within_{r}'] = (linf <= r).float().mean().item()
        else:
            metrics['click_l2'] = 0.0
            for r in click_radii:
                metrics[f'click_within_{r}'] = 0.0
    else:
        metrics['acc_click'] = 0.0
        metrics['click_l2'] = 0.0
        for r in click_radii:
            metrics[f'click_within_{r}'] = 0.0
    metrics['n_click'] = float(n_click)
    return metrics
