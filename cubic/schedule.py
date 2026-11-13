"""Pure-ALU Cubic Hermite Polynomial Schedule (Smoothstep)."""

from typing import Callable
import jax.numpy as jnp


def cubic_smoothstep(u: jnp.ndarray) -> jnp.ndarray:
    """Exact smoothstep polynomial S(u) = (1 - u)^2 * (1 + 2u)."""
    return (1.0 - u) * (1.0 - u) * (1.0 + 2.0 * u)


def cubic_schedule(
    peak_lr: float = 0.002,
    total_steps: int = 2385,
    warmup_steps: int = 48,
    min_lr_ratio: float = 0.1,
) -> Callable[[int], jnp.ndarray]:
    """Pure ALU Cubic Hermite schedule with linear warmup and Smoothstep decay."""
    floor = peak_lr * min_lr_ratio
    warm = float(warmup_steps)
    T = float(total_steps)

    def fn(step: int) -> jnp.ndarray:
        t = jnp.asarray(step, jnp.float32) + 1.0
        warm_f = jnp.minimum(1.0, t / warm)
        prog = jnp.clip((t - warm) / jnp.maximum(1.0, T - warm), 0.0, 1.0)
        s_u = cubic_smoothstep(prog)
        lr = floor + (peak_lr - floor) * s_u
        return jnp.where(t < warm, peak_lr * warm_f, lr)

    return fn
