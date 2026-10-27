"""Model architecture presets and training configurations for ARDS and Comparison pipelines.

Configures the 4 canonical comparison runs:
    {Standard, Algebraic} x {Cosine, ARDS}
    1. standard_cosine:  Standard Transformer + AdamW + Cosine schedule
    2. algebraic_cosine: Pure Algebraic Transformer + AdamW + Cosine schedule
    3. standard_ards:    Standard Transformer + AdamW + ARDS schedule
    4. algebraic_ards:   Pure Algebraic Transformer + AdamW + ARDS schedule

Enforces strictly:
1. 125M parameter scale (plus proxy for unit tests / smoke).
2. Seed 42 initialization and single-seed training.
3. Proper schedule assignment across all 4 comparison runs.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional

from algebraic.config import SIZES
from algebraic.model import LMConfig

COMPARISON_4_RUNS: List[str] = [
    "standard_cosine",
    "algebraic_cosine",
    "standard_ards",
    "algebraic_ards",
]
FOUR_RUNS = COMPARISON_4_RUNS

VALID_SCHEDULES = (
    "ards",
    "rational",
    "algebraic",
    "rational_decay",
    "algebraic_rational_decay",
    "cosine",
    "cos",
    "cosine_decay",
    "cosine_annealing",
)


def validate_schedule(schedule: Optional[str]) -> str:
    """Validate schedule name, supporting both ARDS and Cosine."""
    if schedule is None:
        return "ards"
    sched_lower = str(schedule).lower().strip()
    if "cos" in sched_lower:
        return "cosine"
    if sched_lower in ("ards", "rational", "algebraic", "rational_decay", "algebraic_rational_decay"):
        return "ards"
    raise ValueError(
        f"Unsupported schedule {schedule!r} in ards/; only 'ards' and 'cosine' are permitted."
    )


def validate_schedule_not_cosine(schedule: Optional[str]) -> str:
    """Strictly validate schedule name; reject cosine schedules with descriptive error."""
    if schedule is None:
        return "ards"
    sched_lower = str(schedule).lower().strip()
    if "cos" in sched_lower:
        raise ValueError(
            "Cosine schedule is strictly prohibited in ards/; "
            "use pure ARDS algebraic rational decay."
        )
    if sched_lower not in ("ards", "rational", "algebraic", "rational_decay", "algebraic_rational_decay"):
        raise ValueError(
            f"Unsupported schedule {schedule!r} in ards/; only 'ards' is permitted."
        )
    return "ards"


def standard_ards(size: str = "125m", **overrides: Any) -> LMConfig:
    """125M Standard Transformer preset with ARDS configuration.

    Learned RMS gains, SwiGLU, RoPE, softmax attention, cross-entropy loss,
    and pure XLA dense execution.
    """
    if "schedule" in overrides:
        validate_schedule_not_cosine(overrides.pop("schedule"))
    if "min_lr_ratio" in overrides:
        raise ValueError("min_lr_ratio is cosine-specific and prohibited in ards/")

    if size not in SIZES:
        raise KeyError(f"unknown size {size!r}; available: {list(SIZES.keys())}")

    cfg = LMConfig(
        **SIZES[size],
        seq_len=2048,
        norm="rms",
        ffn="swiglu",
        pos="rope",
        attn="softmax",
        sink="none",
        head="ce",
        qk_norm=False,
        attn_impl="dense",
        flash_block=512,
        head_impl="dense",
        head_block_n=128,
        head_block_v=512,
        dtype="bfloat16",
    )
    return replace(cfg, **overrides)


def algebraic_ards(size: str = "125m", **overrides: Any) -> LMConfig:
    """125M Pure Algebraic Transformer preset with ARDS configuration.

    Learned RMS gains, ALU-GLU (beta=0.5), geometric Cayley rotations,
    QK RMS normalization, Asinh-power attention (n=16, c=8.0) with
    learned sink logit, Algebraic likelihood head (n=8, c=2.0),
    and pure XLA dense execution.
    """
    if "schedule" in overrides:
        validate_schedule_not_cosine(overrides.pop("schedule"))
    if "min_lr_ratio" in overrides:
        raise ValueError("min_lr_ratio is cosine-specific and prohibited in ards/")

    if size not in SIZES:
        raise KeyError(f"unknown size {size!r}; available: {list(SIZES.keys())}")

    cfg = LMConfig(
        **SIZES[size],
        seq_len=2048,
        norm="rms",
        ffn="alu_glu",
        alu_beta=0.5,
        pos="cayley_geo",
        qk_norm=True,
        attn="asinh",
        attn_n=16,
        attn_c=8.0,
        sink="learned",
        sink_init=0.0,
        head="alg_ce",
        head_n=8,
        head_c=2.0,
        attn_impl="dense",
        flash_block=512,
        head_impl="dense",
        head_block_n=128,
        head_block_v=512,
        dtype="bfloat16",
    )
    return replace(cfg, **overrides)


def standard_cosine(size: str = "125m", **overrides: Any) -> LMConfig:
    """125M Standard Transformer preset configured for Cosine decay schedule.

    Learned RMS gains, SwiGLU, RoPE, softmax attention, cross-entropy loss,
    and pure XLA dense execution.
    """
    overrides.pop("schedule", None)
    overrides.pop("min_lr_ratio", None)

    if size not in SIZES:
        raise KeyError(f"unknown size {size!r}; available: {list(SIZES.keys())}")

    cfg = LMConfig(
        **SIZES[size],
        seq_len=2048,
        norm="rms",
        ffn="swiglu",
        pos="rope",
        attn="softmax",
        sink="none",
        head="ce",
        qk_norm=False,
        attn_impl="dense",
        flash_block=512,
        head_impl="dense",
        head_block_n=128,
        head_block_v=512,
        dtype="bfloat16",
    )
    return replace(cfg, **overrides)


def algebraic_cosine(size: str = "125m", **overrides: Any) -> LMConfig:
    """125M Pure Algebraic Transformer preset configured for Cosine decay schedule.

    Learned RMS gains, ALU-GLU (beta=0.5), geometric Cayley rotations,
    QK RMS normalization, Asinh-power attention (n=16, c=8.0) with
    learned sink logit, Algebraic likelihood head (n=8, c=2.0),
    and pure XLA dense execution.
    """
    overrides.pop("schedule", None)
    overrides.pop("min_lr_ratio", None)

    if size not in SIZES:
        raise KeyError(f"unknown size {size!r}; available: {list(SIZES.keys())}")

    cfg = LMConfig(
        **SIZES[size],
        seq_len=2048,
        norm="rms",
        ffn="alu_glu",
        alu_beta=0.5,
        pos="cayley_geo",
        qk_norm=True,
        attn="asinh",
        attn_n=16,
        attn_c=8.0,
        sink="learned",
        sink_init=0.0,
        head="alg_ce",
        head_n=8,
        head_c=2.0,
        attn_impl="dense",
        flash_block=512,
        head_impl="dense",
        head_block_n=128,
        head_block_v=512,
        dtype="bfloat16",
    )
    return replace(cfg, **overrides)

    cfg = LMConfig(
        **SIZES[size],
        seq_len=2048,
        norm="rms",
        ffn="alu_glu",
        alu_beta=0.5,
        pos="cayley_geo",
        qk_norm=True,
        attn="asinh",
        attn_n=16,
        attn_c=8.0,
        sink="learned",
        sink_init=0.0,
        head="alg_ce",
        head_n=8,
        head_c=2.0,
        attn_impl="flash",
        flash_block=512,
        head_impl="fused",
        head_block_n=128,
        head_block_v=512,
        dtype="bfloat16",
    )
    return replace(cfg, **overrides)


ARDS_PRESETS: Dict[str, Callable[..., LMConfig]] = {
    "standard_ards": standard_ards,
    "algebraic_ards": algebraic_ards,
    "standard_cosine": standard_cosine,
    "algebraic_cosine": algebraic_cosine,
    "standard": standard_ards,
    "algebraic": algebraic_ards,
}

COMPARISON_PRESETS = ARDS_PRESETS


def get_ards_config(preset: str, size: str = "125m", **overrides: Any) -> LMConfig:
    """Lookup and construct an LMConfig for a comparison model preset."""
    if preset not in ARDS_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; available: {list(ARDS_PRESETS.keys())}")
    return ARDS_PRESETS[preset](size, **overrides)


@dataclass(frozen=True)
class ARDSTrainingConfig:
    """Complete training hyperparameters for a 125M comparison pretraining run.

    Supports the 4 canonical runs:
    - standard_ards
    - algebraic_ards
    - standard_cosine
    - algebraic_cosine
    """

    # Model architecture
    arm: str = "standard_ards"
    size: str = "125m"
    seq_len: int = 2048

    # Optimization & Token Budget
    tokens: int = 2_500_000_000
    batch: int = 512
    micro: int = 4
    seed: int = 42
    data_seed: int = 42

    # Schedule parameters
    peak_lr: float = 3e-3
    warmup_steps: int = 48
    total_steps: int = 2385
    schedule: str = "ards"
    min_lr_ratio: float = 0.1

    # AdamW hyperparameters
    b1: float = 0.9
    b2: float = 0.95
    adam_eps: float = 1e-8
    wd: float = 0.1
    clip: float = 1.0

    # Execution
    attn_impl: str = "dense"
    head_impl: str = "dense"
    flash_block: int = 512
    head_block_n: int = 128
    head_block_v: int = 512

    # Data & Logging
    train_file: str = "fineweb_train_2_5B.npy"
    valid_file: str = "fineweb_valid.npy"
    eval_tokens: int = 8_000_000
    eval_batch: int = 4
    eval_every: int = 250
    log_every: int = 25
    ckpt_every: int = 200
    keep_ckpts: int = 2

    def __post_init__(self):
        sched_lower = str(self.schedule).lower().strip()
        if self.arm in ("standard_ards", "algebraic_ards") and "cos" in sched_lower:
            raise ValueError(
                f"Cosine schedule is strictly prohibited in ards/; "
                f"use pure ARDS algebraic rational decay, or arm='standard_cosine' / 'algebraic_cosine'."
            )
        validate_schedule(self.schedule)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_lm_config(self, **overrides: Any) -> LMConfig:
        return get_ards_config(
            self.arm,
            size=self.size,
            seq_len=self.seq_len,
            attn_impl=self.attn_impl,
            head_impl=self.head_impl,
            flash_block=self.flash_block,
            head_block_n=self.head_block_n,
            head_block_v=self.head_block_v,
            **overrides,
        )


def get_training_config(arm: str = "standard_ards", **overrides: Any) -> ARDSTrainingConfig:
    """Factory creating an ARDSTrainingConfig for one of the 4 comparison runs."""
    if arm in ("standard_cosine", "algebraic_cosine") and "schedule" not in overrides:
        overrides["schedule"] = "cosine"
    elif arm in ("standard_ards", "algebraic_ards") and "schedule" in overrides:
        validate_schedule_not_cosine(overrides["schedule"])
    return ARDSTrainingConfig(arm=arm, **overrides)

