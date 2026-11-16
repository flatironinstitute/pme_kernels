"""Tables 5.1 and 5.2 of the paper from the record of the SI kernel study (design/records/review_si_coarsest_record.json).

Table 5.1: the smallest order P <= 10 whose SI kernel meets the tolerance on the grid that ESP uses, s = 1, for B_P
and phi_P^* in the limit of large boxes and the psi_0 kernel in a box of side 10 r_c; tolerances that no kernel of the
study meets on this grid are left out. Table 5.2: the largest grid spacing h/r_c = pi/(c_s s*) for P = 4, 5, 6.
Writes paper/table_5_1.tex and paper/table_5_2.tex.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent/'design'/'records'/'review_si_coarsest_record.json'


def power(x):
    return f'$10^{{{int(round(math.log10(x)))}}}$'


def spacing(c, s):
    return '--' if s is None else f'${math.pi/(c*s):.3g}$'


def main():
    results = {(r['tolerance'], r['P']): r for r in json.loads(RECORD.read_text())['results']}
    tolerances = sorted({t for t, _ in results}, reverse=True)
    orders = sorted({P for _, P in results})
    lines = [r'\begin{tabular}{@{}lccc@{}}', r'\toprule', r'$\varepsilon$ & $B_P$ & $\psi_0$ & $\phi_P^*$\\', r'\midrule']
    for t in tolerances:
        cells = []
        for key in ('spline_sigma', 'prolate_sigma', 'sigma'):
            found = [P for P in orders if results[t, P][key] is not None and results[t, P][key] <= 1+1e-9]
            cells.append(str(min(found)) if found else f'$>{max(orders)}$')
        if any(not c.startswith('$>') for c in cells):
            lines.append(' & '.join([power(t)]+cells)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (HERE/'table_5_1.tex').write_text('\n'.join(lines)+'\n')
    shown = (4, 5, 6)
    lines = [r'\begin{tabular}{@{}lc'+'ccc'*len(shown)+r'@{}}', r'\toprule',
             ' & & '+' & '.join(rf'\multicolumn{{3}}{{c}}{{$P={P}$}}' for P in shown)+r'\\',
             ''.join(rf'\cmidrule(lr){{{3+3*i}-{5+3*i}}}' for i in range(len(shown))),
             r'$\varepsilon$ & $c_s$'+r' & $B_P$ & $\psi_0$ & $\phi_P^*$'*len(shown)+r'\\', r'\midrule']
    for t in tolerances:
        c = results[t, shown[0]]['c_s']
        cells = [power(t), f'${c:.2f}$']
        for P in shown:
            r = results[t, P]
            cells += [spacing(c, r['spline_sigma']), spacing(c, r['prolate_sigma']), spacing(c, r['sigma'])]
        lines.append(' & '.join(cells)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (HERE/'table_5_2.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
    for name in ('table_5_1.tex', 'table_5_2.tex'):
        print((HERE/name).read_text())
