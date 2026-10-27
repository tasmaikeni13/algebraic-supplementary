"""ARDS and 4-Run Comparison Pipelines and Configurations.

This module provides dedicated, single-seed (seed 42) training and evaluation
pipelines for the 4 comparison runs:
    {Standard, Algebraic} x {Cosine, ARDS}
    1. standard_cosine:  Standard Transformer + AdamW + Cosine schedule
    2. algebraic_cosine: Pure Algebraic Transformer + AdamW + Cosine schedule
    3. standard_ards:    Standard Transformer + AdamW + ARDS rational decay
    4. algebraic_ards:   Pure Algebraic Transformer + AdamW + ARDS rational decay
"""

from ards.configs import (
    ARDS_PRESETS,
    ARDSTrainingConfig,
    COMPARISON_4_RUNS,
    COMPARISON_PRESETS,
    FOUR_RUNS,
    algebraic_ards,
    algebraic_cosine,
    get_ards_config,
    get_training_config,
    standard_ards,
    standard_cosine,
    validate_schedule,
    validate_schedule_not_cosine,
)
from ards.evaluate import (
    evaluate_ards_run,
    evaluate_main,
    evaluate_recall,
    evaluate_zero_shot,
    run_ards_evaluation,
    validate_ards_checkpoint_purity,
)
from ards.schedule import (
    DEFAULT_125M_PEAK_LR,
    DEFAULT_125M_TOTAL_STEPS,
    DEFAULT_125M_WARMUP_STEPS,
    DEFAULT_SCALE_FACTOR,
    FORBIDDEN_COSINE_KEYWORDS,
    PERMITTED_ARDS_NAMES,
    ards_learning_rate,
    ards_schedule,
    ards_schedule_with_warmup,
    cosine_schedule,
    get_schedule,
    rational_decay_factor,
    reject_cosine_schedule,
    verify_schedule_purity,
)
from ards.train import (
    SyntheticWindowSampler,
    latest_checkpoint,
    main as train_main,
    run_ards_training,
    save_checkpoint,
)

__all__ = [
    # Schedule
    "DEFAULT_125M_PEAK_LR",
    "DEFAULT_125M_TOTAL_STEPS",
    "DEFAULT_125M_WARMUP_STEPS",
    "DEFAULT_SCALE_FACTOR",
    "FORBIDDEN_COSINE_KEYWORDS",
    "PERMITTED_ARDS_NAMES",
    "ards_learning_rate",
    "ards_schedule",
    "ards_schedule_with_warmup",
    "cosine_schedule",
    "get_schedule",
    "rational_decay_factor",
    "reject_cosine_schedule",
    "validate_schedule",
    "validate_schedule_not_cosine",
    "verify_schedule_purity",
    # Configurations & Presets (4 Runs)
    "standard_ards",
    "algebraic_ards",
    "standard_cosine",
    "algebraic_cosine",
    "COMPARISON_4_RUNS",
    "COMPARISON_PRESETS",
    "FOUR_RUNS",
    "ARDS_PRESETS",
    "get_ards_config",
    "ARDSTrainingConfig",
    "get_training_config",
    # Training Pipeline
    "SyntheticWindowSampler",
    "latest_checkpoint",
    "run_ards_training",
    "save_checkpoint",
    "train_main",
    # Evaluation Pipeline
    "evaluate_ards_run",
    "evaluate_main",
    "evaluate_recall",
    "evaluate_zero_shot",
    "run_ards_evaluation",
    "validate_ards_checkpoint_purity",
]
