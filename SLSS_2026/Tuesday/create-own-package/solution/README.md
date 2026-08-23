# spectra

Find peaks in 1-D spectra. This is the **reference solution** for the packaging workshop.

## Install

```bash
pip install spectra-demo
```

## Use it as a library

```python
from spectra import load_csv, moving_average, find_peaks

x, y = load_csv("../start/data/sample_spectrum.csv")
peaks = find_peaks(x, moving_average(y, window=5), threshold=4.0)
for p in peaks:
    print(p.position, p.height)
```

## Use it from the shell

```bash
spectra-find ../start/data/sample_spectrum.csv --window 5 --threshold 4.0
```

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
