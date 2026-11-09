"""WARC world model: a transformer over the frame history whose action enters as an adaLN condition, plus a learned
vector added to the patch token of the clicked cell.

The backbone is UnifiedHistWM (vendored, unchanged). The click enters there only as a global condition
(type embedding + concat(coord_x, coord_y), the adaLN condition of every block). This model keeps that and also adds
a learned vector, one per cell of the 2x2 patch (zero-initialised), to the patch token that contains the clicked cell
of the current action, right after the patch embedding.

Token index: _patch_tokens flattens the (g, g) patch grid row-major, so a click at (x = column, y = row) lands on
token (y // p) * g + (x // p) and on cell (y % p) * p + (x % p) of that patch. Augmented clicks arrive already
transformed (transform_action6_coord), so the marker follows the frame as the global embedding does.
"""
import torch
from torch import nn

from unified_wm.src.model import (ARCH_NAME as BACKBONE_ARCH, NUM_INPUT_COLORS, UNIFIED_DEFAULTS,
                              UnifiedHistWM)

ARCH_NAME = 'unified_hist_cm'


class WarcWM(UnifiedHistWM):
    """UnifiedHistWM whose clicked cell's patch token also carries a learned marker."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # (p*p, D): one vector per cell of a patch; zero so that the untrained model is the backbone
        self.click_embed = nn.Parameter(torch.zeros(self.patch_size ** 2, self.embed_dim))

    def click_token(self, action_xs, action_ys):
        """(valid, token index, cell index) of each sample's click; index 0 and valid = False without a click."""
        valid = (action_xs >= 0) & (action_ys >= 0)
        xs, ys = action_xs.clamp(min=0).long(), action_ys.clamp(min=0).long()
        p, g = self.patch_size, self.patch_grid
        return valid, (ys // p) * g + (xs // p), (ys % p) * p + (xs % p)

    def add_click_embed(self, x, action_xs, action_ys):
        """x (B, Np, D) patch tokens -> the same with click_embed[cell] added to the clicked token of each sample
        that has a click. Out of place (index_put with accumulate), so DDP and torch.compile see a plain op."""
        if action_xs is None or action_ys is None:
            return x
        valid, idx, sub = self.click_token(action_xs, action_ys)
        add = self.click_embed[sub] * valid.to(self.click_embed.dtype).unsqueeze(-1)
        ar = torch.arange(x.size(0), device=x.device)
        return x.index_put((ar, idx), add.to(x.dtype), accumulate=True)

    def forward_dynamics(self, pixel_values, task_ids, action_ids, action_xs=None, action_ys=None,
                         attention_mask=None, hist_obs=None, hist_action_ids=None, hist_action_xs=None,
                         hist_action_ys=None):
        """UnifiedHistWM.forward_dynamics with the marker added to the patch tokens before their dropout."""
        if attention_mask is not None:
            raise ValueError('unified_hist does not take an attention mask')
        if pixel_values.dim() != 3:
            raise ValueError('`pixel_values` must be (batch, 64, 64).')
        if hist_obs is None:
            hist_obs, hist_action_ids, hist_action_xs, hist_action_ys = self._default_window(pixel_values)
        x = self._patch_tokens(hist_obs)
        x = self.add_click_embed(x, action_xs, action_ys)
        x = self.dropout(x)
        c = self._build_condition(action_ids, action_xs, action_ys, hist_action_ids, hist_action_xs,
                                  hist_action_ys)
        for blk in self.blocks:
            x = blk(x, c)
        return self._ffp_logits(self.norm(x))

    def forward(self, *args, **kwargs):
        return {'ffp_logits': self.forward_dynamics(*args, **kwargs)}


def build_model(*, hist_n=3, patch_size=2, device=None, **overrides):
    """build_unified_model with WarcWM: the same cfg resolution, _reset_parameters and marker at zero."""
    cfg = dict(UNIFIED_DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    model = WarcWM(hist_n=hist_n, patch_size=patch_size, **cfg)
    if device is not None:
        model = model.to(device)
    resolved = dict(cfg, arch=ARCH_NAME, backbone=BACKBONE_ARCH, click_embed=True, hist_n=hist_n,
                    patch_size=patch_size, num_input_colors=NUM_INPUT_COLORS,
                    conditioning='adaln_zero_all_layers + learned vector on the clicked patch token (zero-init)',
                    norm='pre', positional='rope_only', task_token=False, attention='sdpa')
    return model, resolved


BUILDERS = {ARCH_NAME: build_model}


def build_from_checkpoint(ckpt_path, *, device):
    """Model and metadata from a checkpoint (arch unified_hist_cm)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = dict(ckpt['config']['model_config'])
    if mc.get('arch') not in BUILDERS:
        raise ValueError(f'not a WARC checkpoint (arch={mc.get("arch")!r})')
    model, _ = BUILDERS[mc['arch']](hist_n=mc['hist_n'], patch_size=mc['patch_size'], device=device,
                                    **{k: mc[k] for k in UNIFIED_DEFAULTS if k in mc})
    model.load_state_dict({k.replace('_orig_mod.', '', 1): v for k, v in ckpt['model_state'].items()})
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != 'model_state'}
    meta['arch'] = mc['arch']
    return model, meta
