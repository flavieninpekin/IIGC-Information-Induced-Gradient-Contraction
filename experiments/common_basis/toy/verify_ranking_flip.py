"""Verify the ranking-preservation theorem for compact kappa measurement.

Theory (notes/variance_decomp_theory.md): for two conditions i, j,

  sign(f_i(N) - f_j(N)) = sign((a_i b_j - a_j b_i) N + (a_i c_j - a_j c_i))

is LINEAR in N, so the compact-measurement ranking either holds for all N
or flips exactly once at N* = (a_j c_i - a_i c_j)/(a_i b_j - a_j b_i).

We verify:
  1. Synthetic counterexample: predicted N* == numerically measured crossing.
  2. Real reveal-sweep components: is the across-level ranking at small N
     the same as the large-N (true) ranking?
"""
import os, json
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                       'data', 'kappa', 'variance_decomp')
os.makedirs(OUT_DIR, exist_ok=True)


def f(N, a, b, c):
    return a / (a + b + c / N)


def predict_crossing(i, j):
    a_i, b_i, c_i = i
    a_j, b_j, c_j = j
    denom = a_i * b_j - a_j * b_i
    if denom == 0:
        return None
    return (a_j * c_i - a_i * c_j) / denom


def main():
    results = {}

    # ---- 1. Synthetic counterexample (theorem's example) ----
    i = (1.0, 0.0, 100.0)   # high kappa, high noise
    j = (1.0, 1.0, 1.0)     # lower kappa, low noise
    Ns = [1, 5, 10, 50, 90, 99, 100, 110, 200, 1000, 10000]
    Ns = sorted(set(Ns))
    print('=== Synthetic: i=(a=1,b=0,c=100), j=(a=1,b=1,c=1) ===')
    print(f'{"N":>7} {"f_i":>8} {"f_j":>8} {"f_i>f_j?":>8}')
    flip_at = None
    prev = None
    for N in Ns:
        fi, fj = f(N, *i), f(N, *j)
        cur = fi > fj
        if prev is not None and cur != prev:
            flip_at = N
        prev = cur
        print(f'{N:>7} {fi:>8.4f} {fj:>8.4f} {str(cur):>8}')
    pred = predict_crossing(i, j)
    print(f'predicted N* = {pred:.1f}')
    print(f'measured flip between N={flip_at if flip_at else "none"}')
    results['synthetic'] = {'predicted_N_star': pred, 'observed_flip': flip_at}

    # ---- 2. Real reveal-sweep components ----
    # from data/kappa/variance_decomp/results.json experiment C
    with open(os.path.join(OUT_DIR, 'results.json')) as fp:
        rd = json.load(fp)
    C = rd['C']
    print('\n=== Real reveal-sweep: rank of f(N) vs kappa_true ===')
    print(f'{"p":>5} {"kappa_true":>10} {"f(10)":>8} {"f(100)":>8} '
          f'{"rank_t":>7} {"rank_10":>7} {"rank_100":>7}')
    rows = []
    for p in ['0.0', '0.25', '0.5', '0.75', '1.0']:
        d = C[p]
        a, b, c = d['E_shared'], d['E_contrast'], d['sigma2']
        kt = a / (a + b) if a + b > 0 else 0.0
        f10 = f(10, a, b, c)
        f100 = f(100, a, b, c)
        rows.append({'p': p, 'kt': kt, 'f10': f10, 'f100': f100})
    rows.sort(key=lambda r: r['kt'])
    for idx, r in enumerate(rows):
        rank_t = idx
        rank_10 = sum(1 for s in rows if s['f10'] < r['f10'])
        rank_100 = sum(1 for s in rows if s['f100'] < r['f100'])
        print(f'{r["p"]:>5} {r["kt"]:>10.4f} {r["f10"]:>8.4f} '
              f'{r["f100"]:>8.4f} {rank_t:>7} {rank_10:>7} {rank_100:>7}')
    results['reveal_sweep'] = rows
    # Kendall tau between kappa_true rank and f(10)/f(100) rank
    kts = [r['kt'] for r in rows]
    f10s = [r['f10'] for r in rows]
    f100s = [r['f100'] for r in rows]
    tau10 = _kendall(kts, f10s)
    tau100 = _kendall(kts, f100s)
    print(f'Kendall tau: rank(kappa_true) vs rank(f(10)) = {tau10:.2f}; '
          f'vs rank(f(100)) = {tau100:.2f}')
    results['kendall'] = {'N10': tau10, 'N100': tau100}

    with open(os.path.join(OUT_DIR, 'ranking_flip.json'), 'w') as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f'\nSaved: {os.path.join(OUT_DIR, "ranking_flip.json")}')


def _kendall(x, y):
    # count concordant/discordant pairs
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx == sy:
                conc += 1
            elif sx != 0 and sy != 0:
                disc += 1
    if conc + disc == 0:
        return 1.0
    return (conc - disc) / (conc + disc)


if __name__ == '__main__':
    main()
