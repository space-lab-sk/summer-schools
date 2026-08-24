import argparse

from rich.console import Console
from rich.table import Table

from spectra import __version__, find_peaks, load_csv, moving_average


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spectra-find", description="Find peaks in a two-column CSV spectrum."
    )
    parser.add_argument("path", help="CSV file with columns (x, y) and a header row")
    parser.add_argument("--window", type=int, default=5, help="smoothing window")
    parser.add_argument("--threshold", type=float, default=0.0, help="min peak height")
    parser.add_argument("--version", action="version", version=f"spectra {__version__}")
    args = parser.parse_args(argv)

    x, y = load_csv(args.path)
    smoothed = moving_average(y, window=args.window)
    peaks = find_peaks(x, smoothed, threshold=args.threshold)

    table = Table(title=f"{len(peaks)} peaks in {args.path} ({len(x)} points)")
    table.add_column("#", justify="right")
    table.add_column("position", justify="right")
    table.add_column("height", justify="right")
    for i, p in enumerate(peaks, start=1):
        table.add_row(str(i), f"{p.position:.3f}", f"{p.height:.3f}")
    Console().print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
