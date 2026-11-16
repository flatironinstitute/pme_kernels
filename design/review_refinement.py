"""Generate the retuned pair-optimal SI comparison in pme_kernels.tex.

Dependencies: Python 3 and NumPy. Run from any directory. The source and
cell integrals use independent Legendre quadratures. Alias powers are
integrated as squared cell defects, so there is no alias truncation or
subtraction of nearly equal powers. The physical reference is the fixed
spherical band of the c=12.024 compact Coulomb split, not the full field.
"""
import json
import csv
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import legval, leggauss
from functools import lru_cache

@lru_cache(None)
def coeff(c,K=40):
    k=np.arange(0,2*K,2,dtype=float)
    d=k*(k+1)+c*c*(2*k*(k+1)-1)/((2*k+3)*(2*k-1))
    off=c*c*(k[:-1]+2)*(k[:-1]+1)/((2*k[:-1]+3)*np.sqrt((2*k[:-1]+1)*(2*k[:-1]+5)))
    A=np.diag(d)+np.diag(off,1)+np.diag(off,-1)
    eig,v=np.linalg.eigh(A)
    co=np.zeros(2*K); co[::2]=v[:,0]*np.sqrt(k+.5)
    if legval(0,co)<0: co=-co
    return co

class Cell:
    def __init__(self,P,q,nq=80):
        self.P=P; self.q=np.asarray(q)
        t,w=leggauss(nq); self.t=t; self.w=w
        # Split at each support boundary inside one grid cell.
        cuts=[0.,.5,1.] if P%2 else [0.,1.]
        self.u=np.concatenate([(b-a)/2*t+(a+b)/2 for a,b in zip(cuts[:-1],cuts[1:])])
        self.wu=np.concatenate([(b-a)/2*w for a,b in zip(cuts[:-1],cuts[1:])])
        ell=np.arange(-P//2-1,P//2+3)
        self.x=ell[:,None]-self.u[None,:]
        self.ins=np.abs(self.x)<=P/2
        self.xleg=np.clip(2*self.x/P,-1,1)
        self.phase=np.exp(1j*self.q[:,None,None]*self.x[None,:,:])
        self.cos=np.cos(P/2*self.q[:,None]*t[None,:])
    def alias(self,c):
        co=coeff(float(c)); z=2*co[0]
        g=legval(self.xleg,co)*self.ins/(self.P/2*z)
        f=self.cos@(self.w*legval(self.t,co))/z
        R=np.einsum('qlu,lu->qu',self.phase,g)
        A=(np.abs(R-f[:,None])**2)@self.wu/f**2
        return A

c=12.024; rc=1.; L=10.; dxi=2*np.pi/L
ks=np.arange(-int(np.floor(c/dxi)),int(np.floor(c/dxi))+1)
K1,K2,K3=np.meshgrid(ks,ks,ks,indexing='ij')
xi=dxi*np.sqrt(K1*K1+K2*K2+K3*K3); mask=(xi>0)&(xi<=c)
co=coeff(c); Sh=np.zeros_like(xi)
Sh[mask]=legval(xi[mask]/c,co)/legval(0.,co)/xi[mask]**2
W=(xi*Sh)**2; W=W[mask]; W=W/W.sum()
ix=[(K[mask]+len(ks)//2) for K in (K1,K2,K3)]
h0=L/40  # An integer periodic base mesh, shared by accuracy and work models.

def score(A):
    logs=np.log1p(A)
    lt=logs[ix[0]]+logs[ix[1]]+logs[ix[2]]
    return np.sqrt(W@np.expm1(2*lt)), np.sqrt(W@(-np.expm1(-2*lt)))

def golden(fun,a,b,steps=28):
    z=(np.sqrt(5)-1)/2
    x=b-z*(b-a); y=a+z*(b-a); fx=fun(x); fy=fun(y)
    for _ in range(steps):
        if fx<fy: b,y,fy=y,x,fx; x=b-z*(b-a); fx=fun(x)
        else: a,x,fx=x,y,fy; y=a+z*(b-a); fy=fun(y)
    return (x,fx) if fx<fy else (y,fy)

def minimize(fun,grid):
    vals=np.array([fun(t) for t in grid]); candidates=[(grid[0],vals[0]),(grid[-1],vals[-1])]
    for j in range(1,len(grid)-1):
        if vals[j]<=vals[j-1] and vals[j]<=vals[j+1]:
            candidates.append(golden(fun,grid[j-1],grid[j+1]))
    return min(candidates,key=lambda x:x[1])


def convolution_alias(q,nq=80):
    from numpy.polynomial.legendre import legint,legmul
    cc=Cell(4,q,nq); co=coeff(5.); z=2*co[0]
    F=legint(co); Ft=legint(legmul(co,[0,1]))
    y=cc.x; a=np.maximum(-1,y-1); b=np.minimum(1,y+1)
    t=np.minimum(b,np.maximum(a,y))
    a=np.clip(a,-1,1); b=np.clip(b,-1,1); t=np.clip(t,-1,1)
    g=((1-y)*(legval(t,F)-legval(a,F))+(legval(t,Ft)-legval(a,Ft))
       +(1+y)*(legval(b,F)-legval(t,F))-(legval(b,Ft)-legval(t,Ft)))/z
    g*=cc.ins
    f=np.sinc(np.asarray(q)/(2*np.pi))**2*(np.cos(np.asarray(q)[:,None]*cc.t)@(cc.w*legval(cc.t,co)))/z
    R=np.einsum('qlu,lu->qu',cc.phase,g)
    return (np.abs(R-f[:,None])**2)@cc.wu/f**2

def bspline_alias(q,P):
    ms=np.r_[np.arange(-400,0),np.arange(1,401)]
    return np.sum((q[:,None]/(q[:,None]+2*np.pi*ms))**(2*P),axis=1)

def one_rms(A):
    return np.sqrt(-np.expm1(-2*np.log1p(A)))

if __name__=='__main__':
    dest=Path(__file__).resolve().parent/'records'
    print('Pair-optimal influence, exact cell aliases, fixed spherical Coulomb reference',flush=True)
    records=[]; worst_quad=0.
    for P in (4,5,6):
        best1=None
        for scale in (1,2,4,8):
            q=h0*dxi*ks/scale; cell=Cell(P,q)
            bs=score(bspline_alias(q,P))[1]
            fun=lambda cw:score(cell.alias(cw))[1]
            cw,val=minimize(fun,np.linspace(0,1.3*np.pi*P,125))
            if scale==1:best1=cw
            fixed=fun(best1)
            natural=score(cell.alias(cw))[0]
            doubled=score(Cell(P,q,160).alias(cw))[1]
            err=abs(doubled/val-1); worst_quad=max(worst_quad,err)
            assert err<1e-7
            hybrid=score(convolution_alias(q))[1] if P==4 else float('nan')
            rec=dict(P=P,s=scale,B=bs,pro=val,cw=cw,natural=natural,fixed=fixed,hybrid=hybrid,quad_rel=err)
            records.append(rec)
            print(rec,flush=True)
    with (dest/'review_retuned_refinement.csv').open('w') as f:
        out=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");out.writeheader();out.writerows(records)
    one=[]
    for scale in 2.**np.arange(7):
        q=np.array([np.pi/(2*scale)]); cell=Cell(4,q)
        cw,val=minimize(lambda z:one_rms(cell.alias(z))[0],np.linspace(0,16.4,160))
        one.append(dict(s=int(scale),B=float(one_rms(bspline_alias(q,4))[0]),
                        fixed=float(one_rms(cell.alias(9.))[0]),pro=val,cw=cw,
                        hybrid=float(one_rms(convolution_alias(q))[0])))
    with (dest/'review_one_mode.csv').open('w') as f:
        out=csv.DictWriter(f,fieldnames=list(one[0]),lineterminator="\n");out.writeheader();out.writerows(one)
    # This is an independent check against the existing MATLAB cell calculation.
    A9=Cell(4,np.array([0.])).alias(9.)[0]
    assert abs(A9/6.123156076510836e-8-1)<1e-7
    metadata=dict(split_c=c,rc=rc,L=L,h0=h0,base_grid_width=40,physical_reference='fixed spherical |xi|<=c/rc; zero mode removed',
                  observable='spectral-force pair RMS, normalized by retained exact force power',
                  influence='pair-optimal for the truncated target, independently for every window',
                  bandwidth_search=[0,1.3*np.pi],bandwidth_search_units='P',
                  cell_quadrature=80,verification_quadrature=160,pswf_even_basis_size=40,
                  maximum_quad_relative_difference=worst_quad,one_mode_cw9_floor_alias=A9)
    (dest/'review_refinement_record.json').write_text(json.dumps(metadata,indent=2)+'\n')
