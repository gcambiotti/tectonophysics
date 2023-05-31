import numpy as np
import matplotlib.pyplot as plt

from pyshtools.legendre import PlmBar, PlBar, legendre_lm, legendre
from pyshtools import SHCoeffs
from pyshtools.shio import SHCilmToVector


dtr = np.pi/180

ind_plm = lambda n,m: n*(n+1)//2 + abs(m)
ind_ylm = lambda n,m: n*(n+1) + m 


def inv_ylm(y):
    n = np.sqrt(y)
    if type(y) == np.ndarray:
        n = n.astype(int)
    else:
        n = int(n)
    m = y - n*(n+1)
    return n,m

def inv_plm(p):
    n = np.sqrt(1/4+2*p)-0.5
    if type(p) == np.ndarray:
        n = n.astype(int)
    else:
        n = int(n)
    m = p - n*(n+1)//2
    return n,m
    

####################################################################
def eva_Ylm(nmax, colat, lon):
        
    plm_max = ind_plm(nmax,nmax)+1
    ylm_max = ind_ylm(nmax,nmax)+1
    
    Plm = PlmBar(nmax,np.cos(colat))
    ms = np.arange(0,nmax+1)

    if type(lon) == np.ndarray:
        
        lonms = np.tensordot(lon,ms,axes=0)
        cos = np.cos(lonms)
        sin = np.sin(lonms)
        #cos[:,1:] *= np.sqrt(2)
        #sin[:,1:] *= np.sqrt(2)
        Ylm = np.empty(lon.shape + (ylm_max,))
        for n in range(nmax+1):
            m = ms[:n+1]
            P = Plm[ind_plm(n,m)]
            ind_y = ind_ylm(n,m)
            Ylm[:,ind_y] = sin[:,m] * P
            ind_y = ind_ylm(n,-m)
            Ylm[:,ind_y] = cos[:,m] * P
            
    else:
        
        lonms = lon*ms
        cos = np.cos(lonms) 
        sin = np.sin(lonms)
        #cos[1:] *= np.sqrt(2)
        #sin[1:] *= np.sqrt(2)
        Ylm = np.empty(ylm_max)
        for n in range(nmax+1):
            m = ms[:n+1]
            P = Plm[ind_plm(n,m)]
            ind_y = ind_ylm(n,m)
            Ylm[ind_y] = sin[m] * P
            ind_y = ind_ylm(n,-m)
            Ylm[ind_y] = cos[m] * P
        
    return Ylm
####################################################################



       
       
####################################################################
def eva_Ylm_grid(nmax, colats, lons):

    ylm_max = ind_ylm(nmax, nmax)+1
    Ylm = np.zeros(colats.shape + lons.shape + (ylm_max,))
    for i,colat in enumerate(colats):
        Ylm[i] = eva_Ylm(nmax, colat, lons)
        
    return Ylm
####################################################################

####################################################################
def SH_to_vect(SH):
    
    nmax = SH.lmax
    ylm_max = ind_ylm(nmax, nmax)+1

    vect = np.empty(ylm_max)
    
    for i in range(2):
        for n in range(nmax+1):
            for m in range(n+1):
                coef = SH.coeffs[i,n,m]
                if i == 0: m = -m
                if m == 0 and i == 1: continue
                vect[ind_ylm(n,m)] = coef
    
    return vect
####################################################################

####################################################################
def eva_Pn(nmax,x,normalized=True):
    """
    Return the (unnormalized) Legendre polynomial up to harmonic
    degree nmax for the given argument x
    """
    
    Pn = np.empty(nmax+1)
    if normalized:
        for n in range(nmax+1):
            Pn[n] = legendre_lm(n,0,x)        
    else:
        for n in range(nmax+1):
            Pn[n] = legendre_lm(n,0,x,normalization="unnorm")
        
    return Pn
####################################################################


####################################################################
def eva_Pnm(nmax,x):
    """
    Return the (unnormalized) associate Legendre polynomial up to harmonic
    degree nmax for the given argument x
    """
            
    Pnm = legendre(nmax,x,normalization="unnorm")
        
    return Pnm
####################################################################


####################################################################
def eva_SH_cap(nmax, cap, lon=90, lat=0, norm=False):
    
    phi = lon*dtr
    theta = (90-lat)*dtr

    x = np.cos(cap*dtr)
    Pn = eva_Pn(nmax+1,x)
    Ylm =  eva_Ylm(nmax,theta,phi)

    ns = np.arange(nmax+1)
    fn = ( x * Pn[1:-1]/np.sqrt(2*ns[1:]+1) - Pn[2:]/np.sqrt(2*ns[1:]+3) ) / (2*ns[1:])
    fn = np.append( (1-x)/2, fn )
    
    if norm: fn /= fn[0]

    for n in range(nmax+1):
        m = np.arange(-n,n+1)
        Ylm[ind_ylm(n,m)] *= fn[n]
    
    return Ylm
####################################################################
