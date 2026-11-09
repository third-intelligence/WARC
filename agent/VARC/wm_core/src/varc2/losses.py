from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from varc2.constants import CLICK_ACTION_ID, IGNORE_INDEX, NUM_ACTIONS

def build_joint_logits(type_logits: torch.Tensor, click_logits: torch.Tensor) -> torch.Tensor:
    B = type_logits.size(0)
    return torch.cat([type_logits, click_logits.reshape(B, -1)], dim=-1)

def action_to_joint_index(action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, num_actions: int, H: int, W: int, click_id: int=CLICK_ACTION_ID) -> torch.Tensor:
    num_non_click = num_actions - 1
    non_click_idx = torch.where(action_id < click_id, action_id, action_id - 1)
    click_idx = num_non_click + action_y.clamp(min=0) * W + action_x.clamp(min=0)
    return torch.where(action_id == click_id, click_idx, non_click_idx).long()

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

def ffp_loss(ffp_logits: torch.Tensor, targets: torch.Tensor, ignore_index: int=IGNORE_INDEX) -> torch.Tensor:
    num_colors = ffp_logits.size(1)
    logits_flat = ffp_logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
    total = F.cross_entropy(logits_flat, targets.reshape(-1), ignore_index=ignore_index, reduction='sum')
    return total / (targets != ignore_index).sum().clamp(min=1)

def bc_joint_loss(type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, *, num_actions: int=NUM_ACTIONS, click_id: int=CLICK_ACTION_ID) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    H, W = (click_logits.shape[-2], click_logits.shape[-1])
    joint_logits = build_joint_logits(type_logits, click_logits)
    joint_target = action_to_joint_index(action_id, action_x, action_y, num_actions, H, W, click_id=click_id)
    per = F.cross_entropy(joint_logits, joint_target, reduction='none')
    loss = per.mean()
    with torch.no_grad():
        is_click = (action_id == click_id).to(per.dtype)
        n_click = is_click.sum()
        n_non_click = per.numel() - n_click
        parts = {'loss_bc': loss.detach(), 'loss_type': (per * (1.0 - is_click)).sum() / n_non_click.clamp(min=1), 'loss_click': (per * is_click).sum() / n_click.clamp(min=1), 'n_click': n_click}
    return (loss, parts)

def policy_wm_loss(ffp_logits: torch.Tensor, targets: torch.Tensor, type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, *, value: Optional[torch.Tensor]=None, value_target: Optional[torch.Tensor]=None, lambda_policy: float=1.0, lambda_dyn: float=1.0, lambda_value: float=1.0, num_actions: int=NUM_ACTIONS, ignore_index: int=IGNORE_INDEX) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    l_dyn = ffp_loss(ffp_logits, targets, ignore_index=ignore_index)
    l_pol, bc_parts = bc_joint_loss(type_logits, click_logits, action_id, action_x, action_y, num_actions=num_actions)
    loss = lambda_policy * l_pol + lambda_dyn * l_dyn
    parts = {'loss_dyn': l_dyn.detach(), 'loss_policy': bc_parts['loss_bc'], 'loss_policy_type': bc_parts['loss_type'], 'loss_policy_click': bc_parts['loss_click']}
    if value is not None and value_target is not None:
        l_val = F.mse_loss(value.float(), value_target.float())
        loss = loss + lambda_value * l_val
        parts['loss_value'] = l_val.detach()
    parts['loss_total'] = loss.detach()
    return (loss, parts)

@torch.no_grad()
def ffp_metrics(ffp_logits: torch.Tensor, targets: torch.Tensor, ignore_index: int=IGNORE_INDEX) -> Dict[str, float]:
    pred = ffp_logits.argmax(dim=1)
    valid = targets != ignore_index
    correct = ((pred == targets) & valid).sum().item()
    total = valid.sum().item()
    B = targets.size(0)
    mismatch = ((pred != targets) & valid).flatten(1).any(dim=1)
    has_valid = valid.flatten(1).any(dim=1)
    exact = (~mismatch & has_valid).sum().item()
    return {'acc_pixel': correct / max(total, 1), 'acc_exact': exact / max(B, 1)}

@torch.no_grad()
def value_metrics(value: torch.Tensor, value_target: torch.Tensor) -> Dict[str, float]:
    v, t = (value.float(), value_target.float())
    return {'value_mse': F.mse_loss(v, t).item(), 'value_mae': (v - t).abs().mean().item()}

@torch.no_grad()
def bc_metrics(type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, *, click_radii: Tuple[int, ...]=(1, 3, 5), num_actions: int=NUM_ACTIONS, click_id: int=CLICK_ACTION_ID) -> Dict[str, float]:
    B, H, W = click_logits.shape
    joint_logits = build_joint_logits(type_logits, click_logits)
    joint_target = action_to_joint_index(action_id, action_x, action_y, num_actions, H, W, click_id=click_id)
    metrics: Dict[str, float] = {}
    pred_idx = joint_logits.argmax(dim=-1)
    pred_action_id, pred_x, pred_y = joint_index_to_action(pred_idx, num_actions, H, W, click_id=click_id)
    metrics['acc_type'] = (pred_action_id == action_id).float().mean().item()
    metrics['acc_joint'] = (pred_idx == joint_target).float().mean().item()
    logp = F.log_softmax(joint_logits, dim=-1)
    metrics['entropy_joint'] = -(logp.exp() * logp).sum(dim=-1).mean().item()
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
            px, py = (pred_x_clk[pred_is_click], pred_y_clk[pred_is_click])
            tx, ty = (true_x[pred_is_click], true_y[pred_is_click])
            metrics['click_l2'] = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2).mean().item()
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
