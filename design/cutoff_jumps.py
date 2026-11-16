"""Jumps at the cutoff of the psi_0 splits, for the bandwidths that ESP uses.

For each tolerance eps in the table of recommended parameters of ESP (Liang
et al., Nature Commun. 17, 6681, 2026) and its splitting bandwidth c, the Coulomb
pair force jumps at r_c by 2 chi(1) times the bare force, where
chi = psi_0^c / int psi_0^c, and the local part of the 1/r^6 split with the
profile w = psi_0^c / psi_0^c(0) jumps by R_6(1^-) = w - 5w'/8 + w''/8 at t=1
times the bare pair energy. NumPy only.
"""
import json
from pathlib import Path

from numpy.polynomial.legendre import legder, legval

import refinement as ref

DEST = Path(__file__).resolve().parent/'records'
ESP_TABLE = [(1e-3, 9.5392), (5e-4, 10.290), (1e-4, 12.024), (5e-5, 12.762), (1e-5, 14.471)]


def main():
    rows = []
    for eps, c in ESP_TABLE:
        co = ref.coeff(c)
        chi, w = co/(2*co[0]), co/legval(0., co)
        jump = 2*legval(1., chi)
        r6 = legval(1., w)-5/8*legval(1., legder(w))+1/8*legval(1., legder(w, 2))
        rows.append(dict(tolerance=eps, c=c, coulomb_force_jump=jump, coulomb_force_jump_over_tolerance=jump/eps,
                         r6_energy_jump=r6, r6_energy_jump_over_tolerance=r6/eps))
        print(rows[-1])
    (DEST/'cutoff_jumps.json').write_text(json.dumps(dict(
        definition='coulomb_force_jump = 2 chi(1), relative to the bare force at r_c; '
                   'r6_energy_jump = R_6(1^-), relative to the bare pair energy at r_c',
        rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
