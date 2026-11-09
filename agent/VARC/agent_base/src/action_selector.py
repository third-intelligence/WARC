from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
CLICK_ACTION_ID = 6

@dataclass(frozen=True)
class Action:
    action_id: int
    x: int = -1
    y: int = -1

def expand_avail_to_joint(avail_mask: torch.Tensor, H: int, W: int, click_id: int=CLICK_ACTION_ID) -> torch.Tensor:
    num_actions = avail_mask.size(-1)
    num_non_click = num_actions - 1
    joint_dim = num_non_click + H * W
    if avail_mask.dim() == 1:
        out = torch.zeros(joint_dim, dtype=torch.bool, device=avail_mask.device)
        for a in range(num_actions):
            if a == click_id:
                continue
            j = a if a < click_id else a - 1
            out[j] = avail_mask[a]
        if bool(avail_mask[click_id]):
            out[num_non_click:] = True
        return out
    B = avail_mask.size(0)
    out = torch.zeros(B, joint_dim, dtype=torch.bool, device=avail_mask.device)
    for a in range(num_actions):
        if a == click_id:
            continue
        j = a if a < click_id else a - 1
        out[:, j] = avail_mask[:, a]
    click_avail = avail_mask[:, click_id].unsqueeze(-1).expand(B, H * W)
    out[:, num_non_click:] = click_avail
    return out

def available_mask_from_list(available_actions: Optional[Sequence[int]], num_actions: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    if not available_actions:
        mask[:] = True
        return mask
    for a in available_actions:
        if 0 <= int(a) < num_actions:
            mask[int(a)] = True
    return mask

def apply_joint_mask(joint_logits: torch.Tensor, joint_avail_mask: torch.Tensor) -> torch.Tensor:
    return joint_logits.masked_fill(~joint_avail_mask.bool(), float('-inf'))

def apply_available_mask(type_logits: torch.Tensor, avail_mask: torch.Tensor) -> torch.Tensor:
    if avail_mask.dim() == 1 and type_logits.dim() == 2:
        avail_mask = avail_mask.unsqueeze(0).expand_as(type_logits)
    return type_logits.masked_fill(~avail_mask.bool(), float('-inf'))

def select_action(type_logits: torch.Tensor, click_logits: torch.Tensor, available_actions: Optional[Sequence[int]]=None, *, sample: bool=False, temperature: float=1.0, num_actions: int=8, click_id: int=CLICK_ACTION_ID) -> Action:
    H, W = (click_logits.shape[-2], click_logits.shape[-1])
    device = type_logits.device
    num_non_click = num_actions - 1
    joint = torch.cat([type_logits, click_logits.reshape(-1)], dim=-1)
    avail = available_mask_from_list(available_actions, num_actions, device)
    joint_avail = expand_avail_to_joint(avail, H, W, click_id=click_id)
    masked = apply_joint_mask(joint, joint_avail)
    if sample:
        probs = F.softmax(masked / max(temperature, 1e-06), dim=-1)
        idx = int(torch.multinomial(probs, num_samples=1).item())
    else:
        idx = int(masked.argmax(dim=-1).item())
    if idx < num_non_click:
        action_id = idx if idx < click_id else idx + 1
        return Action(action_id=int(action_id), x=-1, y=-1)
    off = idx - num_non_click
    y = off // W
    x = off % W
    return Action(action_id=int(click_id), x=int(x), y=int(y))

def joint_entropy(type_logits: torch.Tensor, click_logits: torch.Tensor, joint_avail_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    B = type_logits.size(0)
    joint = torch.cat([type_logits, click_logits.reshape(B, -1)], dim=-1)
    if joint_avail_mask is not None:
        joint = apply_joint_mask(joint, joint_avail_mask)
    logp = F.log_softmax(joint, dim=-1)
    p = logp.exp()
    if joint_avail_mask is not None:
        ent_terms = -(p * logp).masked_fill(~joint_avail_mask.bool(), 0.0)
    else:
        ent_terms = -(p * logp)
    return ent_terms.sum(dim=-1)

def type_entropy(type_logits: torch.Tensor, avail_mask: torch.Tensor) -> torch.Tensor:
    masked = apply_available_mask(type_logits, avail_mask)
    logp = F.log_softmax(masked, dim=-1)
    p = logp.exp()
    ent = -(p * logp).masked_fill(~avail_mask.bool(), 0.0).sum(dim=-1)
    return ent
