"""Deterministic seed derivation, so every random draw in a season traces back to CONFIG["SEED"].

With OPTUNA_N_JOBS > 1 trial completion order can still vary between runs.
"""
import zlib

import numpy as np


def derive_seed(base_seed: int, *parts) -> int:
    """Combine base_seed with extra keys (location, year, counter, ...) into a reproducible 32-bit seed."""
    entropy = [int(base_seed)] + [zlib.crc32(str(p).encode()) for p in parts]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])
