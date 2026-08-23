import csv
from pathlib import Path


def load_csv(path: str | Path) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y
