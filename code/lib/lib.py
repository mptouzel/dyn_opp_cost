import numpy as np
import math
import pandas as pd
from itertools import product
from scipy.interpolate import griddata
para=dict()
para['T']=15
para['T_ITI']=para['T']/2
para['p']=1/2

def get_trajs():
    df_traj=pd.DataFrame(columns=['seq'])
    df_traj.seq=pd.Series(product([-1,1],repeat=15))
    df_traj.seq=df_traj.seq.apply(lambda x:np.asarray(x))
    df_traj.head()
    df_traj['Nt']=df_traj.seq.apply(lambda x: np.insert(np.cumsum(x),0,0))
    df_traj['nCorrectChoice']=(df_traj['Nt'].apply(lambda x:x[-1])>0)
    return df_traj 

def get_survprob(df_data,Nt_samples,data_boundary,T=para['T']):
    tb,b_bel=data_boundary[1]
    b_beltmp=[]
    for t in range(T+1):
        teff=t-(T-len(b_bel))-1
        b_beltmp.append(b_bel[teff] if teff>=0 else 1)
    b_beltmp=np.asarray(b_beltmp)
    df_data['tDecision']=df_data.Pt_plus.apply(lambda x: np.where(np.logical_or(x>=b_beltmp,x<=1-b_beltmp))[0][0] if np.where(np.logical_or(x>=b_beltmp,x<=1-b_beltmp))[0].size else T)

    #state occupancy count distributions in (N_p,N_m) space
    
    tdec_vec=df_data.tDecision.apply(lambda x:int(x)).values
    surv_prob=np.zeros((T+1,T+1))
    for Np in np.arange(T+1):
        for Nm in np.arange(T+1):
            if Np+Nm<=T and Np+Nm>0:#=0
                surv_prob[Np,Nm]=np.sum(tdec_vec[Nt_samples[:,Np+Nm-1]==Np-Nm]>Np+Nm)/np.sum(Nt_samples[:,Np+Nm-1]==Np-Nm) 
    surv_prob[0,0]=1  
    surv_prob[np.isnan(surv_prob)]=0
    
    ###plot on smoothed coordinates.
    mesh=np.meshgrid(range(T+1),range(T+1))
    Npvec=mesh[0].flatten()
    Nmvec=mesh[1].flatten()
    tvec=Npvec+Nmvec
    Nvec=Npvec-Nmvec
    tvecdense=np.linspace(min(tvec),max(tvec),100)
    Nvecdense=np.linspace(min(Nvec),max(Nvec),100)
    z_d=griddata((tvec,Nvec),surv_prob.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    z_d[z_d<0]=0    
    return (tvecdense,Nvecdense,z_d)

def binomial(n, k):    
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def get_pt_plus(t,Nt,T=para['T'],p=para['p']):
    '''
    Outputs probability that state>=0 at time T if state is Nt at time t, for +ve jump probability, p.
    Behaves strangely for T a multiple of 10...
    '''
    if t==-1:
        return p
    else:
        tp=T - t #-1
        Nt_plus=(t+Nt)/2.
        if tp<(T-1)/2.-(t-Nt)/2.:
            if Nt>0:
                return 1
            else:
                return 0
        else:
            #Cisek;'s'
            NL=(t-Nt)/2.
            Nc=T-t
            kvec=np.arange(0,(np.min((Nc,(T-1)/2-NL))+1))
            return np.power(p,Nc)*np.sum(np.asarray([binomial(int(Nc), int(k)) for k in kvec]))
#             if p<=0.5:
#                 kvec=np.arange(0,2*(np.min((Nc,(T-1.)/2.-NL))+1),2)
#                 return np.power(p,Nc)*np.sum(np.asarray([binomial(int(Nc), int(k/2.)) for k in kvec]))
#             else:

#                 kvec=np.arange(np.min((Nc,(T-1.)/2.-NL)))
#                 return np.power(p,Nc)*np.sum(np.asarray([binomial(int(Nc), int(k)) for k in kvec]))
#             return np.power(p,Nc)*np.sum(np.asarray([binomial(int(Nc), int(k/2.)) for k in kvec]))
            #mine:
#             kvec=np.arange(0,(T-1)/2.+Nt_plus+1)
# #             return np.sum( np.asarray([binomial(int(tp), int(k)) for k in kvec])* np.power(p,kvec) * np.power(1 - p,tp-kvec))
#             return np.sum( special.binom(tp,kvec)* np.power(p,kvec) * np.power(1 - p,tp-kvec))
#             kvec=np.arange(0,tp+1)
#             return np.sum( np.asarray([binomial(int(tp), int(k)) for k in kvec])* np.power(p,kvec) * np.power(1 - p,tp-kvec))
#             return np.power(p,tp-Nt)*np.sum( special.binom(tp,kvec))



#data loading
#-NHP
#-HP
#-RL

#strategy solving
#-dyn prog framework
#-data save


#strategy diagnostics
#-compute survival density
#-compute decesion density

