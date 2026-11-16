/* Example and check of pme_kernels.h: load the table of phi_5^* for the tolerance 1e-4 and print the stencil of a
 * particle, with the sums that charge and dipole conservation make one and zero.
 *
 *   cc -O2 -o example include/example.c -lm && ./example data/tables/phi_P5_eps1e-04.txt 5 0.3
 */
#include "pme_kernels.h"

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "data/tables/phi_P5_eps1e-04.txt";
    int P = argc > 2 ? atoi(argv[2]) : 5, i, j0;
    double x = argc > 3 ? atof(argv[3]) : 0.3, w[64], dw[64], charge = 0., dipole = 0.;
    pmk_table T;
    if (pmk_table_load(path, P, &T)) { fprintf(stderr, "cannot read %s as a table of order %d\n", path, P); return 1; }
    j0 = pmk_stencil(&T, x, w, dw);
    for (i = 0; i < P; i++) {
        printf("%d %.17g %.17g\n", j0+i, w[i], dw[i]);
        charge += w[i];
        dipole += (j0+i-x)*w[i];
    }
    printf("sum of weights - 1 = %.3e, dipole = %.3e\n", charge-1., dipole);
    pmk_table_free(&T);
    return 0;
}
