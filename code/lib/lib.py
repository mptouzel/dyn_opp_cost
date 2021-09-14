import numpy as np
import math
import pandas as pd
from itertools import product
from scipy.interpolate import griddata
para=dict()
if False:
    para['T']=15
    para['T_ITI']=8#7.5 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
    para['p']=1/2
    para['tp']=0
else:
    para['T']=11
    para['T_ITI']=1 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
    para['p']=1/2
    para['tp']=0

def get_trajs(T=para['T']):
    df_traj=pd.DataFrame(columns=['seq'])
    df_traj.seq=pd.Series(product([-1,1],repeat=T))
    df_traj.seq=df_traj.seq.apply(lambda x:np.asarray(x))
    df_traj['Nt']=df_traj.seq.apply(lambda x: np.insert(np.cumsum(x),0,0))
    df_traj['nCorrectChoice']=(df_traj['Nt'].apply(lambda x:x[-1])>0)
    #def dummy(Nt_seq):
        #return np.array([(get_pt_plus(t,Nt) if Nt>=0 else 1-get_pt_plus(t,Nt)) for t,Nt in enumerate(Nt_seq)])
    def dummy(Nt_seq):
        return np.array([get_pt_plus(t,Nt)  for t,Nt in enumerate(Nt_seq)])
    df_traj['p_plus']=df_traj.Nt.apply(lambda x: dummy(x))
    #def dummy(Nt_seq):
        #return np.array([(get_pt_plus(t,Nt)  for t,Nt in enumerate(Nt_seq)])
    df_traj['p_success']=df_traj.p_plus.apply(lambda x: np.asarray([max([xel,1-xel]) for xel in x]))
    return df_traj 

def get_survprob(df_data,Nt_samples,data_boundary=None,T=para['T']):
    if data_boundary is not None:
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
    
    return surv_prob
    ####plot on smoothed coordinates.
    #mesh=np.meshgrid(range(T+1),range(T+1))
    #Npvec=mesh[0].flatten()
    #Nmvec=mesh[1].flatten()
    #tvec=Npvec+Nmvec
    #Nvec=Npvec-Nmvec
    #tvecdense=np.linspace(min(tvec),max(tvec),100)
    #Nvecdense=np.linspace(min(Nvec),max(Nvec),100)
    #z_d=griddata((tvec,Nvec),surv_prob.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    #z_d[z_d<0]=0    
    #return (tvecdense,Nvecdense,z_d)

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
    assert t>=0, "time must be non-negative"
    t=int(t)
    Nt_plus=(t+Nt)/2.
    lower_bound=np.ceil(T/2-Nt_plus)
    if lower_bound>0:
        kvec=np.arange(lower_bound,T-t+1,dtype=int)
        #return 2**(-(T-t)+np.log2(np.sum( [binomial(T - t,k) for k in kvec])))
        return np.power(p,T-t)*np.sum( [binomial(T - t,k) for k in kvec])
    else:
        return 1

def get_trial_duration(t,alpha):
    return t+(1-alpha)*(para['T']-t)+para['T_ITI']