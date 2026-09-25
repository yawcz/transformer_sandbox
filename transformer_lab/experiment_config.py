MODEL_CONFIG = {
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,
    "rope_theta": 10000,
    "num_layers": 4,
    "num_heads": 16,
}

OPTIMIZER_CONFIG = {
    "a_max": 1e-3,
    "a_min": 1e-3 / 100,
    "T_w": int(0.05 * 50000),
    "T_c": 50000,
    "betas": (0.9, 0.999),
    "weight_decay": 0.01,
    "max_l2_norm": 1.0,
}

TRAIN_CONFIG = {
    "num_iters": 50000,
    "num_valid_iters": 50,
    "eval_steps": 500,
    "batch_size": 52,
}

PROBE_CONFIG = {
    "random_seed": 42,
    "repeat_length": 50,
    "num_prompts": 10,
    "sample_lower": 256,
    "sample_upper": 3000,
}
