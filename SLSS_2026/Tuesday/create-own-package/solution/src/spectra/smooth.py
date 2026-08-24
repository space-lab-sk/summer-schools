

def moving_average(y: list[float], window: int = 5) -> list[float]:
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    half = window // 2
    out = []
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out
