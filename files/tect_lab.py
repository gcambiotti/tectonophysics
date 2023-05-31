import matplotlib.pyplot as plt
import numpy as np


##############################################
dtr = np.pi/180
##############################################



##############################################
def eva_Pn(z,N):          
    """
    \n\n Return the Legendre polynomials evaluated at "x" and up to maximum degree "N"
    """
    Pn = np.empty(N+1)
    Pn[0] = 1
    Pn[1] = z
    for n in range(1,N):
        Pn[n+1] = ( (2*n+1)*z*Pn[n]-n*Pn[n-1] ) / (n+1)
    return Pn
##############################################


##############################################
def eva_Pns(zs,N):
    """
    \n\n Return the Legendre polynomial evaluated at different "zs" and up to the maximum degree "N"
    """
    Pns = np.empty(zs.shape+(N+1,))
    Pns[:,0] = 1
    Pns[:,1] = zs
    for n in range(1,N):
        Pns[:,n+1] = ( (2*n+1)*zs*Pns[:,n]-n*Pns[:,n-1] ) / (n+1)
    return Pns
##############################################

##############################################
def eva_cap(gammas,alpha=30):
    
    cap = np.zeros(gammas.shape)     # cap function
    cap[gammas<alpha] = 1
    
    return cap
##############################################

    
##############################################
def eva_cap_degree(N,alpha=30):

    a = np.cos(alpha*dtr)      # cosine
    Pn = eva_Pn(a,N+1)         # Legendre polynomial at "a" up to maximum degree "N+1"

    fn = np.empty(N+1)        # Legendre coefficients
    fn[0] = (1-a)/2           
    for n in range(1,N+1):
        fn[n] = (1+1/(2*n)) * (a*Pn[n] - Pn[n+1])
        
    return fn
##############################################



##############################################
def eva_shield(gammas,alpha=30):
    
    a = np.cos(alpha*dtr)
    zs = np.cos(gammas)
    shield = np.zeros(zs.shape)
    shield = (zs-a)/(1-a)
    shield[zs<a] = 0    

    return cap
##############################################

    
##############################################
def eva_shield_degree(N,alpha=30):

    a = np.cos(alpha*dtr)      # cosine
    Pn = eva_Pn(a,N+1)         # Legendre polynomial at "a" up to maximum degree "N+1"

    hn = np.empty(N+1)          # Legendre coefficients of the shield function

    hn[0] = (1-a)/4
    hn[1] = (1-a)*(2+a)/4 
    for n in range(2,N+1):
        hn[n] = (2*n+1)/(2*n*(n-1)*(2+n)*(1-a)) * ( ( a**2*(2+n) - n ) * Pn[n] - 2 * a * Pn[n+1]) 

    return hn
##############################################




from pyshtools.legendre import PlmBar as PnmBar

##############################################
def eva_Pnms(N,lats):
    
    P = (N+1)*(N+2)//2

    zs = np.sin(lats*dtr)  #it corresponds to the cosine of the colatitude

    Pnms = np.empty((Nlat,P))
    for i,z in enumerate(zs):
        Pnms[i] = PnmBar(N,z)
        
    return Pnms
##############################################


##############################################
def eva_Ynms(N,lats,lons):

    Pnms = eva_Pnms(N,lats)

    Ynms = np.empty((len(lats),len(lons),P,2))
    
    for n in range(N+1):
        for m in range(n+1):
            p = n*(n+1)//2 + m
            Ynms[:,:,p,0] = np.tensordot(Pnms[:,p], np.cos(m*lons*dtr), axes=0)
            Ynms[:,:,p,1] = np.tensordot(Pnms[:,p], np.sin(m*lons*dtr), axes=0)
            
    return Ynms 
##############################################
