# run it with:  python analysis.py
# (you may have to edit the path below first)

import csv

FILE = "./start/data/sample_spectrum.csv"
WINDOW = 5
THRESHOLD = 4.0

x = []
y = []
f = open(FILE)
r = csv.reader(f)
next(r)
for row in r:
    x.append(float(row[0]))
    y.append(float(row[1]))
f.close()

# smooth it
sm = []
for i in range(len(y)):
    lo = max(0, i - WINDOW // 2)
    hi = min(len(y), i + WINDOW // 2 + 1)
    sm.append(sum(y[lo:hi]) / (hi - lo))

# find peaks
peaks = []
for i in range(1, len(sm) - 1):
    if sm[i] > sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > THRESHOLD:
        peaks.append((x[i], sm[i]))

print("file:", FILE)
print("points:", len(x))
print("peaks found:", len(peaks))
for p in peaks:
    print("  at", round(p[0], 2), "with value:", round(p[1], 3))
