/* pme_kernels.h: evaluate the optimized SI kernels phi_P^* from their tables (header only, C99).
 *
 * A table data/tables/phi_P{P}_eps{eps}.txt lists t, phi(t), phi'(t), phi''(t) at t = k/parts, k = 0, ..., parts P/2,
 * in units of the grid spacing; phi is even and vanishes with its first two derivatives at |t| = P/2. On each part,
 * phi is the polynomial of degree five that matches the three columns at both ends, so the table is twice
 * continuously differentiable, as the kernel is.
 *
 * Spreading a charge q at y (coordinate along one axis, in units of the grid spacing h, x = y/h) assigns to the grid
 * points j0, ..., j0 + P - 1 the one-dimensional weights phi(x - j); the three-dimensional weight is the product of the
 * three. Analytic differentiation uses the derivatives phi'(x - j)/h.
 */
#ifndef PME_KERNELS_H
#define PME_KERNELS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int P;          /* order: grid points per dimension coupled to a particle */
    int parts;      /* parts per grid interval */
    int count;      /* nodes: parts*P/2 + 1 */
    double *f, *df, *d2f;
} pmk_table;

/* Read a table; returns 0 on success. P is the order of the kernel. */
static int pmk_table_load(const char *path, int P, pmk_table *T)
{
    FILE *fp = fopen(path, "r");
    char line[512];
    int cap = 4096, n = 0;
    double t, f, df, d2f, t1 = 0.;
    if (!fp) return 1;
    T->P = P;
    T->f = malloc(cap*sizeof(double));
    T->df = malloc(cap*sizeof(double));
    T->d2f = malloc(cap*sizeof(double));
    if (!T->f || !T->df || !T->d2f) { fclose(fp); return 2; }
    while (fgets(line, sizeof line, fp)) {
        if (line[0] == '#') continue;
        if (sscanf(line, "%lf %lf %lf %lf", &t, &f, &df, &d2f) != 4) continue;
        if (n == cap) {
            cap *= 2;
            T->f = realloc(T->f, cap*sizeof(double));
            T->df = realloc(T->df, cap*sizeof(double));
            T->d2f = realloc(T->d2f, cap*sizeof(double));
            if (!T->f || !T->df || !T->d2f) { fclose(fp); return 2; }
        }
        if (n == 1) t1 = t;
        T->f[n] = f; T->df[n] = df; T->d2f[n] = d2f;
        n++;
    }
    fclose(fp);
    if (n < 2 || t1 <= 0.) return 3;
    T->parts = (int)lround(1./t1);
    T->count = n;
    return (n == T->parts*P/2+1) ? 0 : 4;
}

static void pmk_table_free(pmk_table *T)
{
    free(T->f); free(T->df); free(T->d2f);
    T->f = T->df = T->d2f = NULL;
}

/* phi(t), and phi'(t) in *dphi if dphi is not NULL. */
static inline double pmk_phi(const pmk_table *T, double t, double *dphi)
{
    double x = fabs(t), h = 1./T->parts, u, u2, u3, u4, u5, f0, f1, d0, d1, e0, e1, p;
    int k;
    if (x >= 0.5*T->P) { if (dphi) *dphi = 0.; return 0.; }
    k = (int)(x*T->parts);
    if (k > T->count-2) k = T->count-2;
    u = x*T->parts-k;
    u2 = u*u; u3 = u2*u; u4 = u3*u; u5 = u4*u;
    f0 = T->f[k]; f1 = T->f[k+1];
    d0 = h*T->df[k]; d1 = h*T->df[k+1];
    e0 = h*h*T->d2f[k]; e1 = h*h*T->d2f[k+1];
    p = f0*(1-10*u3+15*u4-6*u5)+d0*(u-6*u3+8*u4-3*u5)+e0*(u2-3*u3+3*u4-u5)/2
        + f1*(10*u3-15*u4+6*u5)+d1*(-4*u3+7*u4-3*u5)+e1*(u3-2*u4+u5)/2;
    if (dphi) {
        double dp = (f0*(-30*u2+60*u3-30*u4)+d0*(1-18*u2+32*u3-15*u4)+e0*(2*u-9*u2+12*u3-5*u4)/2
                     + f1*(30*u2-60*u3+30*u4)+d1*(-12*u2+28*u3-15*u4)+e1*(3*u2-8*u3+5*u4)/2)/h;
        *dphi = (t < 0) ? -dp : dp;
    }
    return p;
}

/* The stencil of a particle at x (grid units): returns j0 and fills w[i] = phi(x - j0 - i) and, if dw is not NULL,
 * dw[i] = phi'(x - j0 - i), for i = 0, ..., P - 1. Grid indices are to be taken modulo the grid size. */
static inline int pmk_stencil(const pmk_table *T, double x, double *w, double *dw)
{
    int j0 = (int)floor(x-0.5*T->P)+1, i;
    for (i = 0; i < T->P; i++)
        w[i] = pmk_phi(T, x-(j0+i), dw ? dw+i : NULL);
    return j0;
}

#endif /* PME_KERNELS_H */
