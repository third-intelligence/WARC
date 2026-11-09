from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from agent_base.src.losses import IGNORE_INDEX_AA3, ffp_loss, bc_joint_loss, ffp_metrics, bc_metrics

def policy_wm_loss(ffp_logits: torch.Tensor, targets: torch.Tensor, type_logits: torch.Tensor, click_logits: torch.Tensor, action_id: torch.Tensor, action_x: torch.Tensor, action_y: torch.Tensor, *, value: Optional[torch.Tensor]=None, value_target: Optional[torch.Tensor]=None, lambda_policy: float=1.0, lambda_dyn: float=1.0, lambda_value: float=1.0, num_actions: int=8, ignore_index: int=IGNORE_INDEX_AA3) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_dyn = ffp_loss(ffp_logits, targets, ignore_index=ignore_index)
    l_pol, bc_parts = bc_joint_loss(type_logits, click_logits, action_id, action_x, action_y, num_actions=num_actions)
    loss = lambda_policy * l_pol + lambda_dyn * l_dyn
    parts = {'loss_dyn': float(l_dyn.detach()), 'loss_policy': bc_parts['loss_bc'], 'loss_policy_type': bc_parts['loss_type'], 'loss_policy_click': bc_parts['loss_click']}
    if value is not None and value_target is not None:
        l_val = F.mse_loss(value, value_target.to(value.dtype))
        loss = loss + lambda_value * l_val
        parts['loss_value'] = float(l_val.detach())
    parts['loss_total'] = float(loss.detach())
    return (loss, parts)
