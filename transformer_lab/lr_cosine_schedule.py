import math

def lr_cosine_schedule(t: int, a_max: float, a_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return (t / T_w) * a_max
    elif T_w <= t and t <= T_c:
        return a_min + (1 / 2) * (1 + math.cos(((t - T_w) / (T_c - T_w)) * math.pi)) * (a_max - a_min)
    else:
        return a_min