from dataclasses import dataclass


@dataclass(frozen=True)
class Peak:
    position: float
    height: float


def find_peaks(x: list[float], y: list[float], threshold: float = 0.0) -> list[Peak]:
    if len(x) != len(y):
        raise ValueError(f"x and y differ in length: {len(x)} vs {len(y)}")
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > threshold:
            peaks.append(Peak(position=x[i], height=y[i]))
    return peaks
