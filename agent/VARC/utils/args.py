import argparse

def resolve_global_batch_size(args, world_size: int, *, distributed: bool=False, rank: int=0) -> None:
    gbs = getattr(args, 'global_batch_size', None)
    if gbs is None:
        return
    if gbs % world_size != 0:
        raise ValueError(f'--global-batch-size={gbs} is not divisible by world_size={world_size}. Choose a multiple of {world_size}.')
    new_bs = gbs // world_size
    if not distributed or rank == 0:
        old_bs = args.batch_size
        print(f'[batch] --global-batch-size={gbs}, world_size={world_size} → per-rank batch_size: {old_bs} → {new_bs}')
    args.batch_size = new_bs

def add_resume_checkpoints(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--resume-skip-task-token', action='store_true', help='Ignore task token embedding weights in the checkpoint when resuming.')
    parser.add_argument('--resume-reset-optimizer', action='store_true', help='Do not load optimizer/scheduler/scaler states when resuming a checkpoint.')
    parser.add_argument('--resume-reset-epoch', action='store_true', help='Ignore the stored epoch in the checkpoint and restart training from epoch 1.')
    parser.add_argument('--init-from-aa12', type=str, default=None, help='Path to an AA1/AA2 checkpoint to use as initialization for AA3 offline training. Only model weights are loaded (optimizer / scheduler / epoch are NOT restored). Keys with shape mismatches are skipped automatically. AA3-specific modules (action_type_embed, action_coord_embed_x/y) are kept at random init since AA1/AA2 checkpoints do not contain them. task_token_embed is also skipped (different num_tasks). Mutually exclusive with --resume-checkpoint (full-resume).')

def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--use-wandb', action='store_true', help='Log training metrics to Weights & Biases.')
    parser.add_argument('--wandb-project', type=str, default='warc', help='Weights & Biases project name.')
    parser.add_argument('--wandb-run-name', type=str, default='', help='Optional run name for Weights & Biases.')
    parser.add_argument('--wandb-resume-id', type=str, default='', help="Existing W&B run id to continue logging into. If set, wandb.init() is called with id=<this>, resume='allow'. Useful for resuming a crashed run so metrics keep the same timeline. Run id can be found in wandb/run-<timestamp>-<id>/ directory names.")

def add_speed_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile optimization (useful for debugging).')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead', choices=('default', 'reduce-overhead', 'max-autotune'), help="torch.compile mode: 'default' (balanced), 'reduce-overhead' (faster), 'max-autotune' (slowest compile, fastest runtime).")
    parser.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision training.')
    parser.add_argument('--amp-dtype', type=str, default='fp16', choices=('fp16', 'bf16'), help="Mixed precision dtype: 'fp16' (default, uses GradScaler) or 'bf16' (wider exponent range, GradScaler not needed; recommended on H100/H200 to avoid NaN from fp16 overflow). Master weights stay fp32 either way.")

def add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank r.')
    parser.add_argument('--lora-alpha', type=float, default=16.0, help='LoRA scaling factor alpha; effective scale = alpha/r.')
    parser.add_argument('--lora-dropout', type=float, default=0.0, help='Dropout probability applied before the LoRA delta.')
    parser.add_argument('--lora-target-modules', type=str, default='qkv,proj', help="Comma-separated attribute names of nn.Linear modules to wrap with LoRA (e.g. 'qkv,proj,linear1,linear2').")
    parser.add_argument('--lora-train-task-token', action='store_true', help='In LoRA mode, also leave task_token_embed trainable.')
    parser.add_argument('--lora-train-action-token', action='store_true', help='In LoRA mode, also leave AA3 action embeddings (action_type_embed / action_coord_embed_x / action_coord_embed_y) trainable.')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train ARC-specific Vision Transformer.')
    add_resume_checkpoints(parser)
    add_wandb_args(parser)
    add_speed_optimizer_args(parser)
    add_lora_args(parser)
    parser.add_argument('--data-root', type=str, default='raw_data/ARC-AGI')
    parser.add_argument('--train-split', dest='train_split', default='training', help='Dataset split to use for training.')
    parser.add_argument('--eval-split', dest='eval_split', default='training', help="Dataset split to evaluate. Use '' to disable evaluation.")
    parser.add_argument('--eval-subset', dest='eval_subset', choices=('train', 'test'), default='test', help='Which example subset to use when evaluating a split.')
    parser.add_argument('--architecture', type=str, choices=('vit', 'unet'), default='vit', help='ViT or UNet architecture')
    parser.add_argument('--unet-size', type=str, choices=('small', 'medium', 'big'), default='medium', help='Choose the size of UNet wanted; small, medium, or big. Only applies if architecture==unet')
    parser.add_argument('--image-size', type=int, default=30)
    parser.add_argument('--num-colors', type=int, default=10)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--mlp-dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32, help='Per-rank batch size (DDP convention).')
    parser.add_argument('--global-batch-size', type=int, default=None, help='Global batch size summed across DDP ranks. When set, --batch-size is automatically overridden to global_batch_size // world_size at runtime, so the global value stays fixed regardless of GPU count. Convergence behavior (lr scaling, num updates per epoch) becomes GPU-count-invariant. Must be divisible by world_size.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num-task-tokens', type=int, default=1, help='Number of task tokens to use in the model.')
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=('none', 'cosine'))
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default=None, help='Optional path to store the final trained checkpoint')
    parser.add_argument('--best-save-path', type=str, default=None, help='Optional path to store the checkpoint achieving the best evaluation accuracy.')
    parser.add_argument('--last-save-path', type=str, default=None, help='Optional path to store the *last-epoch* checkpoint (overwritten every epoch). Includes model_state / optimizer_state / scheduler_state / scaler_state / epoch / best_eval_accuracy / wandb_run_id so the run can be fully resumed after a job timeout. Used together with --resume-checkpoint on restart.')
    parser.add_argument('--ttt-num-each', type=int, default=2)
    parser.add_argument('--vis-every', type=int, default=25)
    parser.add_argument('--patch-size', type=int, default=2)
    parser.add_argument('--include-rearc', action='store_true', help='Add tasks from the Re-ARC dataset to the training set.')
    parser.add_argument('--rearc-path', type=str, default='raw_data/re_arc', help='Path to the Re-ARC dataset root.')
    parser.add_argument('--rearc-limit', type=int, default=-1, help='Maximum number of Re-ARC examples to include per task (use -1 for all examples).')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed data parallel training (torchrun recommended).')
    parser.add_argument('--disable-translation', action='store_true')
    parser.add_argument('--disable-resolution-augmentation', action='store_true')
    parser.add_argument('--fix-scale-factor', type=int, default=1)
    parser.add_argument('--num-attempts', type=int, default=10, help='Number of attempts per evaluation example.')
    parser.add_argument('--eval-save-name', type=str, default=None, help='Name for saving evaluation predictions.')
    parser.add_argument('--aa3-mode', action='store_true', help='Use ARC-AGI-3 trajectory loader (ARCAA3Dataset) instead of the AA1/AA2 loader.')
    parser.add_argument('--num-actions', type=int, default=0, help='Number of action types for action embedding. 0 = disabled (AA1/AA2). Set to 8 for AA3.')
    parser.add_argument('--disable-augment-aa3', action='store_true', help='Disable trajectory-level augmentation in AA3 mode.')
    parser.add_argument('--aa3-task', type=str, default=None, help="Restrict AA3 dataset to a single game/task directory name (e.g. 'ls20'). Only data/<split>/<task>/ is loaded; all other task directories are ignored. Default (None) loads every task directory under the split (current behaviour). Used by TTT scripts where TTT should be focused on one game; offline pretraining typically leaves this unset to learn across all tasks.")
    parser.add_argument('--ttt-max-steps', type=int, default=None, help='Maximum number of training steps (samples) for TTT. Data is taken from the beginning in recording order with no augmentation. Fewer steps than one recording are supported.')
    parser.add_argument('--ttt-eval-every', type=int, default=1, help='how often (in epochs) accuracy and loss on the evaluation data are recorded during TTT; 0 disables it')
    return parser.parse_args()
