# +
from scipy.optimize import root_scalar
from functools import partial
import math
import time
import numpy as np
import pandas as pd
para=dict()
if True:
    para['T']=15
    para['T_ITI']=8#7.5 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
    para['p']=1/2
    para['tp']=0
else:
    para['T']=11
    para['T_ITI']=0 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
    para['p']=1/2
    para['tp']=0
para['move_cost']=0
from copy import deepcopy
from lib.lib import get_pt_plus,get_trial_duration

#plotting
def plot_dynprog_results(choice_value_update,cost_rate_seq,alp,pl):
    
    gamma=1-alp
    g=gamma
    def V_at_t0(rho,choice_value_update,cost_rate_seq,g):
        value=np.zeros((para['T']+1,2*para['T']+1))
        for t in np.arange(para['T']+1)[::-1]:
            for Nt in range(-t,t+1):
                if (t+Nt-1)%2:
                    choice_values=choice_value_update(t,Nt,rho,value,cost_rate_seq[t],g=g)    
                    value[t,para['T']+Nt]=max(choice_values)
        return value[0,para['T']]
    V_at_t0_part=partial(V_at_t0,choice_value_update=choice_value_update,cost_rate_seq=cost_rate_seq,g=gamma)
    
    out_struct=root_scalar(V_at_t0_part,x0=1,method='brentq',bracket=[0,10])
    rho=out_struct.root
    
    #run backup from last to first time step
    value=np.zeros((para['T']+1,2*para['T']+1))
    decide=-np.inf*np.ones((para['T']+1,2*para['T']+1))
    dec_plus=np.zeros(decide.shape)
    dec_minus=np.zeros(decide.shape)
    dec_wait=np.zeros(decide.shape)
    for t in np.arange(para['T']+1)[::-1]:
        for Nt in range(-t,t+1):
            if (t+Nt-1)%2:
                choice_values=choice_value_update(t,Nt,rho,value,cost_rate_seq[t],g=gamma)   
                value[t,para['T']+Nt]=max(choice_values)
                
                if not math.isclose(choice_values[0],choice_values[1]) and not math.isclose(choice_values[2],choice_values[1]):
                    decide[t,para['T']+Nt]=np.argmax(choice_values)-1
                else: #if wait or act of equal value, then act
                    decide[t,para['T']+Nt]=1 if Nt>0 else -1
                dec_plus[t,para['T']+Nt]=choice_values[2]
                dec_minus[t,para['T']+Nt]=choice_values[0]
                dec_wait[t,para['T']+Nt]=choice_values[1]

    fig,ax=pl.subplots(1,4,figsize=(20,5))
    
    #plot values
    for Nm in 2*np.arange(para['T']):
        ax[0].plot(np.arange(Nm/2,para['T']+1-Nm/2)+Nm/2,[np.max([dec_minus[int(t),int(para['T']+Nt)],dec_plus[int(t),int(para['T']+Nt)]]) for t,Nt in zip(np.arange(Nm/2,para['T']+1-Nm/2)+Nm/2,np.arange(Nm/2,para['T']+1-Nm/2)-Nm/2)],'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
        ax[0].plot(np.arange(Nm/2,para['T']+1-Nm/2)+Nm/2,[dec_wait[int(t),int(para['T']+Nt)] for t,Nt in zip(np.arange(Nm/2,para['T']+1-Nm/2)+Nm/2,np.arange(Nm/2,para['T']+1-Nm/2)-Nm/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',clip_on=False)
    for Np in 2*np.arange(para['T']-7):
        ax[0].plot(np.arange(Np/2+1)+Np/2       ,[np.max([dec_minus[int(t),int(para['T']+Nt)],dec_plus[int(t),int(para['T']+Nt)]]) for t,Nt in zip(np.arange(Np/2+1)+Np/2,np.arange(Np/2+1)-Np/2)]              ,'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
        ax[0].plot(np.arange(Np/2+1)+Np/2       ,[dec_wait[int(t),int(para['T']+Nt)] for t,Nt in zip(np.arange(Np/2+1)+Np/2,np.arange(Np/2+1)-Np/2)]              ,'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',clip_on=False)
    ax[0].legend(labels=('act','wait'),frameon=False)
    ax[0].set_xlim(0,15)
    for Nm in 2*np.arange(para['T']):
        ax[1].plot(np.arange(Nm/2,para['T']+1-Nm/2)+Nm/2,[np.max([dec_minus[int(t),int(para['T']+Nt)],dec_plus[int(t),int(para['T']+Nt)]])-dec_wait[int(t),int(para['T']+Nt)] for t,Nt in zip(np.arange(Nm/2,para['T']+1-Nm/2)+Nm/2,np.arange(Nm/2,para['T']+1-Nm/2)-Nm/2)],'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
    for Np in 2*np.arange(para['T']-7):
        ax[1].plot(np.arange(Np/2+1)+Np/2       ,[np.max([dec_minus[int(t),int(para['T']+Nt)],dec_plus[int(t),int(para['T']+Nt)]])-dec_wait[int(t),int(para['T']+Nt)] for t,Nt in zip(np.arange(Np/2+1)+Np/2,np.arange(Np/2+1)-Np/2)]              ,'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
    ax[1].plot([0,15],np.asarray([0,0]),'k--')
    ax[1].set_ylim(-1,1)
    ax[1].set_xlim(0,15)
    ax[1].set_title(r'$V_{\textrm{act}}-V_{\textrm{wait}}$')

    #plot decision boundary (action policy)
    b=[]
    first=True
    double_first=False
    for cit,col in enumerate(decide):
        if np.where(col==1)[0].size:
            first_val=np.where(col==1)[0][0]
            if first:
                startb_t=cit
                b.append(first_val)
                print(para['T']+cit)
                if b[0]<para['T']+cit:
                    print('here')
                    double_first=cit
                first=False
            else:
                b.append(first_val)
    b=np.asarray(b)-para['T']
    tb=np.arange(startb_t,para['T']+1,dtype=int)
    if double_first:
        newvals=np.arange(2,double_first-b[0]+1,2)
        b=np.concatenate((b[0]+newvals,b))
        tb=np.concatenate((tb[0]*np.ones(newvals.shape),tb))

    ax[2].imshow(decide.T,origin='lower',extent=[-0.5,15.5,-15.5,15.5],aspect='auto')#'none', 'nearest', 'bilinear', 'bicubic',
    ax[2].plot(tb,b,'o-')
    ax[2].set_xlim(0,15)
    for Nm in 2*np.arange(para['T']):
        ax[3].plot(np.arange(para['T']+1-Nm/2)+Nm/2,[get_pt_plus(t,Nt) for t,Nt in zip(np.arange(para['T']+1-Nm/2)+Nm/2,np.arange(para['T']+1-Nm/2)-Nm/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for Np in 2*np.arange(para['T']):
        ax[3].plot(np.arange(para['T']+1-Np/2)+Np/2,[1-get_pt_plus(t,Nt) for t,Nt in zip(np.arange(para['T']+1-Np/2)+Np/2,np.arange(para['T']+1-Np/2)-Np/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for tit,t in enumerate(tb):
        if tit==0:
            p=ax[3].plot([t],[get_pt_plus(t,b[tit])],'o')
            col=p[-1].get_color()
        else:
            ax[3].plot([t],[get_pt_plus(t,b[tit])],'o',color=col)
        ax[3].plot([t],[1-get_pt_plus(t,b[tit])],'o',color=col)
        
    fig,ax=pl.subplots()
    for Nm in 2*np.arange(para['T']):
        ax.plot(np.arange(para['T']+1-Nm/2)+Nm/2,[value[int(t),int(para['T']+Nt)]-rho*t for t,Nt in zip(np.arange(para['T']+1-Nm/2)+Nm/2,np.arange(para['T']+1-Nm/2)-Nm/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for Np in 2*np.arange(para['T']):
        ax.plot(np.arange(para['T']+1-Np/2)+Np/2,[value[int(t),int(para['T']+Nt)]-rho*t for t,Nt in zip(np.arange(para['T']+1-Np/2)+Np/2,np.arange(para['T']+1-Np/2)-Np/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')

#     if gamma<1:
#         gamma_vec=np.arange(50)/50
#         bound=bvec[np.argmax(rewardvec_b_vec[np.where(gamma_vec==gamma)[0],:])]
#         ax[3].plot(ax[3].get_xlim(),[bound]*2,'--',color=col)
    
    
    #back out reward rate from reward rate+cost objective: want to see if adding cost actually increases reward rate?
#     tmin=4  
#     constant_cost=-0.07
#     power=1#3
#     cost=0
#     #monotonic
#     cost=constant_cost*(1 if t<tmin else 1-3*(t/para['T'])**power)
#     print(rho+cost)
#     b=np.concatenate(([np.Inf]*startb_t,b))
#     tb=np.concatenate((range(startb_t),tb))
    print(rho)
    return rho,(tb,b)


#DP functions
def V_at_t0(rho,cost_rate_seq,gam,init_cost=0,infer_cost=False,T=para['T'],p=para['p'],move_cost=para['move_cost']):
    if infer_cost:
        data_boundary=cost_rate_seq #overload for data_bounday when inferring cost
        tb,b_bel_seq=data_boundary[1]
        tb,b_evi_seq=data_boundary[0]
        cost_rate_seq=np.zeros(T+1)
    value=np.zeros((T+1,2*T+1))
    for t in np.arange(T+1)[::-1]:
        if infer_cost:
            teff=t-(T+1-len(b_evi_seq))
            cost_rate=get_cost_rate(t,b_evi_seq[teff],b_bel_seq[teff],rho,value) if teff>=0 else 0
            cost_rate_seq[t]=cost_rate
        for Nt in range(-t,t+1):
            if (t+Nt-1)%2:
                choice_values=choice_value_update(t,Nt,rho,value,cost_rate_seq[t],init_cost=init_cost if t==T else 0, g=gam,move_cost=move_cost)    
                value[t,T+Nt]=max(choice_values)
    return value[0,T]

def get_cost_rate(t,Nt_bound,bt_bound,rho,value,T=para['T'],T_ITI=para['T_ITI'],p=para['p'],g=1/4,dt=1,t_penalty=para['tp'],move_cost=para['move_cost']):
    R=np.eye(2)   #2x2 reward matrix: N_T<0,N_T>0 by h_T<0,h_T>0
    Vwait=-np.Inf if t==T else (p*value[t+1,T+Nt_bound+1]+(1-p)*value[t+1,T+Nt_bound-1])
    prob=np.asarray([bt_bound,1-bt_bound]).reshape([1,2]) #prob
    avgR=prob@R
    avgR-=dt*((T-t)*g+T_ITI+(1-prob)*t_penalty)*(rho)-move_cost #subtract integrated reward over remainder of trial      
    return Vwait-dt*rho-np.max(avgR) #inferred cost

def choice_value_update(t,Nt,rho,value,cost_rate,init_cost=0,T=para['T'],T_ITI=para['T_ITI'],p=para['p'],g=1/4,dt=1,t_penalty=para['tp'],move_cost=para['move_cost']):
    gamma=1#move_cost #1 if ~np.isnan(rho) else 0.99
    if np.isnan(rho):
        rho=0
    b=0
    #R=np.asarray([[1+b,0],[0,1-b]])   #reward matrix N_T<0,N_T>0 by h_T<0,h_T>0
    R_corr=1
    R_incorr=move_cost#0#-1.1
    R=np.asarray([[R_corr+b,R_incorr],[R_incorr,R_corr-b]])   #reward matrix N_T<0,N_T>0 by h_T<0,h_T>0
    prob=np.asarray([get_pt_plus(t,Nt),1-get_pt_plus(t,Nt)]).reshape([1,2]) #prob
    avgR=prob@R
    avgR=avgR-(dt*(1+(T-t)*g+T_ITI+(1-prob)*t_penalty)*(rho))#-move_cost#-0.1#+cost_rate) #subtract integrated reward over remainder of trial (even)  
    #Vwait=0 if t==T else gamma*(p*value[t+1,T+Nt+1]+(1-p)*value[t+1,T+Nt-1]) #force action at last time step
    Vwait=-np.Inf if t==T else (p*value[t+1,T+Nt+1]+(1-p)*value[t+1,T+Nt-1])
    Vwait-=dt*(rho+cost_rate) #subtract integrated reward and cost over step
    return [avgR[0,1],Vwait,avgR[0,0]] #choice values

def get_softmax_action(Q_values,beta):
    probs=np.exp(beta*Q_values)
    probs/=np.sum(probs)
    return np.random.choice(len(probs),1, p=probs)-1
    #rng = np.random.default_rng()
    #return rng.choice(len(probs),1, p=probs)-1

def get_action_trajs(state_seq,Q_value_fn,beta=0.1):
    for t,Nt in enumerate(state_seq):
        Q_values=np.asarray([Q_value_fn[0][t,para['T']+Nt],Q_value_fn[1][t,para['T']+Nt],Q_value_fn[2][t,para['T']+Nt]])
        action=get_softmax_action(Q_values,beta)
        decision_data=(t,action)
        if action:
            break
    return decision_data

def get_next_state_inds(state_ind):
    return [(state_ind[0]+1,state_ind[1]),(state_ind[0],state_ind[1]+1)]

def get_psuccess(state_ind,action):
    t=sum(state_ind)
    Nt=2*state_ind[1]-t
    p_plus=get_pt_plus(t,Nt)
    return p_plus if action==1 else 1-p_plus

def get_expected_reward(state_ind,reporting_action,reward_incorrect):
    p=get_psuccess(state_ind,reporting_action)
    reward_correct=1
    return p*reward_correct+(1-p)*reward_incorrect

def get_opt_action(choice_values):
    #greedy action depends on if there is degeneracy
    #if not math.isclose(choice_values[0],choice_values[1]) and not math.isclose(choice_values[2],choice_values[1]):
        #opt_act=np.argmax(choice_values)-1
    #else:
        #opt_act=1 if Nt>0 else -1
        
    args_max=[] #store all actions that acheive the optimal value
    for it,val in enumerate(choice_values):
        if math.isclose(val,max(choice_values)):
            args_max.append(it)                     
    if len(args_max)==1:
        opt_act=np.argmax(choice_values)-1
    elif len(args_max)>1:
        if len(args_max)==2 and (1 in args_max):
            opt_act=args_max[args_max!=1]-1
        else:
            opt_act=1# avoid randomness here       
    return opt_act

def get_policy_from_value(V,state_set,ref_state,reward_incorrect,mode='DR',T=para['T'],gamma=0.99,p=para['p'],terminal_value=0):
    #Q_wait=np.zeros(V.shape)
    #Q_left=np.zeros(V.shape)
    #Q_right=np.zeros(V.shape)
    jump_dist=np.array([1-p,p])
    decide=-np.inf*np.ones((T+1,2*T+1))
    value=np.zeros((T+1,2*T+1))
    if mode=='AR':
        Qref=V[ref_state]
    
    for state_ind in state_set:

        Q_left=get_expected_reward(state_ind,-1,reward_incorrect)
        Q_right=get_expected_reward(state_ind,1,reward_incorrect)
        if np.sum(state_ind)==T:
            Q_wait=terminal_value
        else:
            Vnextvec=np.array([V[ind] for ind in get_next_state_inds(state_ind)])
            avgVnext=np.sum(jump_dist*Vnextvec)
            if mode=='AR':
                Q_wait=0-Qref+avgVnext
            elif mode=='DR':
                Q_wait=0    +gamma*avgVnext
                
        t=sum(state_ind)
        Nt=2*state_ind[1]-t
        choice_values=[Q_left,Q_wait,Q_right]
        
        #greedy action depends on if there is degeneracy
        decide[t,T+Nt]=get_opt_action(choice_values)                     
                            

    return decide
    
def run_value_iteration(mode='AR',threshold=1e-2,max_iter=1e3,gamma=0.99,reward_incorrect=0,seed=0,T=para['T'],p=para['p']):
    
    np.random.seed(seed)
    init_Vval=-0.5
    V=init_Vval*np.ones((T+1,T+1))
    for Nm in range(1,T+1):
        V[Nm,-Nm:]=np.nan
    state_set=[tuple(x) for x in list(np.argwhere(~np.isnan(V)))]
    valid_inds=np.where(~np.isnan(V))
    action_set=np.array([-1,0,1])
    
    #max_diff=np.inf
    Vold=-np.inf*np.ones(V.shape)
    terminal_value=0.5
    jump_dist=np.array([1-p,p])
    #rho=0
    #eta=1e-3
    #diff_rho=np.Inf if mode=='AR' else 0
    ref_state=state_set[0]
    Qref=V[ref_state]
    it=-1
    max_diff_store=[]
    while it<max_iter:#max_diff>threshold and # or diff_rho>threshold:
        it+=1
        for state_ind in np.random.permutation(state_set):
            
            state_ind=tuple(state_ind)
            
            Q=np.zeros((3,))
            for reporting_action in [-1,1]:
                Q[action_set==reporting_action]=get_expected_reward(state_ind,reporting_action,reward_incorrect)
                if mode=='DR':
                    Q[action_set==reporting_action]+=gamma*Qref
            wait_action=0
            R_current=0
            if np.sum(state_ind)==T:
                Q[action_set==wait_action]=R_current + gamma*Qref
            else:
                Vnextvec=np.array([Vold[ind] for ind in get_next_state_inds(state_ind)])
                avgVnext=np.sum(jump_dist*Vnextvec)
               
                if mode=='AR':
                    Q[action_set==wait_action]=R_current-Qref+avgVnext
                elif mode=='DR':
                    Q[action_set==wait_action]=R_current    +gamma*avgVnext

                Q[action_set==wait_action]=np.sum(jump_dist*Vnextvec)
                
            V[state_ind]=np.max(Q)
            
        Qref=V[ref_state]
            #if np.sum(state_ind)!=T:
                #Vnextvec=np.array([Vold[ind] for ind in get_next_state_inds(state_ind)])
                #avgVnext=np.sum(jump_dist*Vnextvec)
                #if np.argmax(Q)==1: #wait
                    #delta=(0-rho+avgVnext)-V[state_ind]
                #else: #act
                    #delta=-rho
                #old_rho=deepcopy(rho)
            #rho+=eta*delta
            #diff_rho=np.fabs(rho-old_rho)
                    
        max_diff=np.max(np.fabs(V[valid_inds]-Vold[valid_inds]))
        max_diff_store.append(max_diff)
        #print(max_diff if mode=='DR' else Qref)
        Vold=deepcopy(V)
        #print(it)
    decide=get_policy_from_value(V,state_set,ref_state,reward_incorrect,gamma=gamma)
    boundary=get_boundary_from_policy(decide)
    
    return V,boundary,max_diff_store
        
def get_DP_boundary(cost_rate_seq,gamma,init_cost=0,infer_cost=False,T=para['T'],move_cost=para['move_cost']):
    
    #root-find reward rate
    #RL_form='discount'
    RL_form='ARRL'
    if RL_form=='ARRL':
        V_at_t0_part=partial(V_at_t0,cost_rate_seq=cost_rate_seq,init_cost=init_cost,infer_cost=infer_cost,gam=gamma,move_cost=move_cost)
        out_struct=root_scalar(V_at_t0_part,x0=1,method='brentq',bracket=[-20,10])
        rho=out_struct.root
    else:
        rho=np.nan
        
    #run backup from last to first time step
    value=np.zeros((T+1,2*T+1))
    decide=-np.inf*np.ones((T+1,2*T+1))
    dec_plus=np.zeros(decide.shape)
    dec_minus=np.zeros(decide.shape)
    dec_wait=np.zeros(decide.shape)
    if infer_cost:
        data_boundary=cost_rate_seq #overload for data_bounday when inferring cost
        tb,b_bel_seq=data_boundary[1]
        tb,b_evi_seq=data_boundary[0]
        cost_rate_seq=np.zeros(T+1)

    for t in np.arange(T+1)[::-1]:
        if infer_cost:
            teff=t-(T+1-len(b_evi_seq))
            cost_rate=get_cost_rate(t,b_evi_seq[teff],b_bel_seq[teff],rho,value,g=gamma) if teff>=0 else 0
            cost_rate_seq[t]=cost_rate
        for Nt in range(-t,t+1):
            if (t+Nt-1)%2:
                choice_values=choice_value_update(t,Nt,rho,value,cost_rate_seq[t],g=gamma,init_cost=init_cost if t==T else 0,move_cost=move_cost)   
                value[t,T+Nt]=max(choice_values) #greedy policy
                    
                decide[t,T+Nt]=get_opt_action(choice_values)    
                dec_plus[t,T+Nt]=choice_values[2]
                dec_minus[t,T+Nt]=choice_values[0]
                dec_wait[t,T+Nt]=choice_values[1]
    
    boundary=get_boundary_from_policy(decide)

    return rho,boundary,(dec_minus,dec_wait,dec_plus),cost_rate_seq,decide

def get_boundary_from_policy(decide,T=para['T']):
    #get decision boundary in evidence space (action policy)
    b=[]
    first=True
    double_first=False
    for cit,col in enumerate(decide):
        if cit==T and b==[]: #if decision at last time then define earliest 100% acc. boundary
            b=np.arange(T+1,T+(T+1)/2+1)[::-1]
            startb_t=(T+1)/2
        else:
            if np.where(col==1)[0].size:
                first_val=np.where(col==1)[0][0]
                if first:
                    startb_t=cit
                    b.append(first_val)
                    if b[0]<T+cit:
                        double_first=cit
                    first=False
                else:
                    b.append(first_val)
    b=np.asarray(b)-T
    tb=np.arange(startb_t,T+1,dtype=int)
    if double_first:
        newvals=np.arange(2,double_first-b[0]+1,2)
        b=np.concatenate((b[0]+newvals,b))
        tb=np.concatenate((tb[0]*np.ones(newvals.shape),tb))
    evi_boundary_series=(tb,b)
    bel_boundary=np.asarray([get_pt_plus(t,Nt) for t,Nt in zip(*evi_boundary_series)])
    return (evi_boundary_series,(tb,bel_boundary))

def get_reward_rate_for_varybound(bel_boundary,dfb,gamma,T=para['T'],T_ITI=para['T_ITI'],output_all=False):
    tb,b_bel=bel_boundary
    if tb[0]>0:
        b_bel=np.concatenate((np.ones(tb[0]),b_bel))
        tb=np.concatenate((np.arange(tb[0]),tb))
    dfb['tb']=dfb.p_plus.apply(lambda x: np.where(np.logical_or(x>=b_bel,x<=1-b_bel))[0][0] if np.where(np.logical_or(x>=b_bel,x<=1-b_bel))[0].size else T)
    dfb['prob_corr_at_tdec']=dfb.apply(lambda row:np.max([row.p_plus[row.tb],1-row.p_plus[row.tb]]),axis=1)   
    meanT=dfb.tb.apply(lambda x: get_trial_duration(x,1-gamma)).mean()
    if not output_all:
        return dfb.prob_corr_at_tdec.mean()/meanT
    else:
        return dfb.prob_corr_at_tdec.mean()/meanT,meanT
    #return (dfb.prob_corr_at_tdec/(dfb.tb+gamma*(T-dfb.tb)+T_ITI)).mean()

def get_decb_from_varyb(bel_boundary,T=para['T']):
    tb,b_bel=bel_boundary
    evi_boundary=[]
    t_vec=[]
    for tind,t in enumerate(tb):
        if np.where([get_pt_plus(t,Nt)>=b_bel[tind] for Nt in np.arange(t%2,t+1,2)])[0].size:
            evi_boundary.append(t%2+2*np.where([get_pt_plus(t,Nt)>=b_bel[tind] for Nt in np.arange(t%2,t+1,2)])[0][0])
            t_vec.append(t)
    evi_boundary_series=(np.asarray(t_vec),np.asarray(evi_boundary))
    bel_boundary=np.asarray([get_pt_plus(t,Nt) for t,Nt in zip(*evi_boundary_series)])
    return evi_boundary_series,(np.asarray(t_vec),bel_boundary)

def get_fixed_boundary(dfb,gamma,T=para['T']):
    bvec=np.linspace(0.5,1,30)
    rewardvec_b_vec=np.asarray([get_reward_rate_for_varybound((np.arange(T+1),ba*np.ones(T+1)),dfb,gamma) for ba in bvec]) #computational bottleneck
    dec_data=[]
    opt_bound=bvec[np.argmax(rewardvec_b_vec)]
    print('opt_b='+str(opt_bound))
    boundary_series=get_decb_from_varyb((np.arange(T+1),opt_bound*np.ones(T+1)))
    return np.max(rewardvec_b_vec),boundary_series,opt_bound,dec_data

def plot_strategy(boundary_series,title_str,cost_space,dec_data=None,T=para['T'],ax=None):

    evi_boundary_series,bel_boundary_series=boundary_series
    tb,b_evi=evi_boundary_series
    tb,b_bel=bel_boundary_series
        
   
    #domain
    min_grayvalue=0.8
    cost_space[cost_space==0]=min_grayvalue
#     for t in range(T+1):
#         cost_space[t,slice(T-t,T+1+t,2)]=0.8
    ax[0].imshow(cost_space.T,origin='lower',cmap='gray',extent=[-0.5,15.5,-15.5,15.5],vmin=0,vmax=1,aspect='auto',)#'none', 'nearest', 'bilinear', 'bicubic',
    
    #decision boundary in evidence space
    ax[0].plot(tb,b_evi,'k.-')
    ax[0].plot(tb,-b_evi,'k.-')
    ax[0].set_xlim(0,15)
    
    #decision boundary in belief space
    for Nm in 2*np.arange(T):
        ax[1].plot(np.arange(T+1-Nm/2)+Nm/2,[get_pt_plus(t,Nt) for t,Nt in zip(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for Np in 2*np.arange(T):
        ax[1].plot(np.arange(T+1-Np/2)+Np/2,[1-get_pt_plus(t,Nt) for t,Nt in zip(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    ax[1].plot(tb,b_bel,'o')
    ax[1].plot(tb,1-b_bel,'o')

    fig.suptitle(title_str)
    
    #plot values
    if len(dec_data)>0:
        dec_minus,dec_wait,dec_plus=dec_data
        fig,ax=pl.subplots(1,2,figsize=(10,5))
        for Nm in 2*np.arange(T):
            ax[0].plot(np.arange(Nm/2,T+1-Nm/2)+Nm/2,[np.max([dec_minus[int(t),int(T+Nt)],dec_plus[int(t),int(T+Nt)]]) for t,Nt in zip(np.arange(Nm/2,T+1-Nm/2)+Nm/2,np.arange(Nm/2,T+1-Nm/2)-Nm/2)],'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
            ax[0].plot(np.arange(Nm/2,T+1-Nm/2)+Nm/2,[dec_wait[int(t),int(T+Nt)] for t,Nt in zip(np.arange(Nm/2,T+1-Nm/2)+Nm/2,np.arange(Nm/2,T+1-Nm/2)-Nm/2)],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',clip_on=False)
        for Np in 2*np.arange(T-7):
            ax[0].plot(np.arange(Np/2+1)+Np/2       ,[np.max([dec_minus[int(t),int(T+Nt)],dec_plus[int(t),int(T+Nt)]]) for t,Nt in zip(np.arange(Np/2+1)+Np/2,np.arange(Np/2+1)-Np/2)]              ,'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
            ax[0].plot(np.arange(Np/2+1)+Np/2       ,[dec_wait[int(t),int(T+Nt)] for t,Nt in zip(np.arange(Np/2+1)+Np/2,np.arange(Np/2+1)-Np/2)]              ,'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',clip_on=False)
        ax[0].legend(labels=('act','wait'),frameon=False)
        ax[0].set_xlim(0,15)
        for Nm in 2*np.arange(T):
            ax[1].plot(np.arange(Nm/2,T+1-Nm/2)+Nm/2,[np.max([dec_minus[int(t),int(T+Nt)],dec_plus[int(t),int(T+Nt)]])-dec_wait[int(t),int(T+Nt)] for t,Nt in zip(np.arange(Nm/2,T+1-Nm/2)+Nm/2,np.arange(Nm/2,T+1-Nm/2)-Nm/2)],'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
        for Np in 2*np.arange(T-7):
            ax[1].plot(np.arange(Np/2+1)+Np/2       ,[np.max([dec_minus[int(t),int(T+Nt)],dec_plus[int(t),int(T+Nt)]])-dec_wait[int(t),int(T+Nt)] for t,Nt in zip(np.arange(Np/2+1)+Np/2,np.arange(Np/2+1)-Np/2)]              ,'ro-',ms=3,lw=0.5,color='r',mew=0.5,mfc='r',mec='r')
        ax[1].plot([0,15],np.asarray([0,0]),'k--')
        ax[1].set_ylim(-1,1)
        ax[1].set_xlim(0,15)
        ax[1].set_title(r'$V_{\textrm{act}}-V_{\textrm{wait}}$')

#general functions
def reward_rate_to_cost(cost_fn,reward_rate_vec,reward_rate_para):
    cost=cost_fn(reward_rate_vec,reward_rate_para)
    return cost

def opp_cost(reward_rate,reward_rate_ref):
    k=12#12
    regret=reward_rate_ref-reward_rate
    return k*regret#-reward_rate_ref

def get_MAP_estimate(reward_rate_dec_time_distr,t):
    ind_t=reward_rate_dec_time_distr.index.get_level_values(0)
    return    (reward_rate_dec_time_distr[ind_t>t].index.get_level_values(1) \
              *reward_rate_dec_time_distr[ind_t>t] \
              /reward_rate_dec_time_distr[ind_t>t].sum() \
              ).sum()

def get_reward_rate_dec_time_distr(boundary,env_df,gamma,T=para['T'],T_ITI=para['T_ITI'],c=0):
    '''
    computes cost function of time from joint distribution 
    of trial-specific reward rate and decision time
    '''
    tb,bv=boundary[0] #evidence space
    num_repeats=np.where(tb[0]==tb[1:])[0].size
    if num_repeats:
        tb=tb[num_repeats:]
        bv=bv[num_repeats:]
#     tb=np.asarray(tb,dtype=int)
#     bv=np.asarray(bv,dtype=int)
    
    env_df['tb_vary']=env_df.Nt.apply(lambda x: int(tb[0] + np.where(np.logical_or(x[int(tb[0]):]>=bv,x[int(tb[0]):]<=1-bv))[0][0] if np.where(np.logical_or(x[int(tb[0]):]>=bv,x[int(tb[0]):]<=1-bv))[0].size else T))
    env_df['prob_corr_at_tdec_vary']=env_df.apply(lambda row:np.max([row.Pt_plus[row.tb_vary],1-row.Pt_plus[row.tb_vary]]),axis=1) 
    env_df['reward_rate_vary']=(env_df.prob_corr_at_tdec_vary-c*env_df.tb_vary)/(env_df.tb_vary+gamma*(T-env_df.tb_vary)+T_ITI)

    return env_df.groupby('tb_vary')['reward_rate_vary'].value_counts()

def solve_opporcost_strategy(knowledge_mode,bound_constr,gamma,plot=False,T=para['T']):

    env_space_df=df_traj.copy() #has Nt sequence and P_+ sequence 

    #solve zero-cost case
    if bound_constr=='none':
        rho_zero,boundary_zero,dec_data,cost_rate_seq=get_DP_boundary(0*np.ones(T+1),gamma)
    elif bound_constr=='fixed':
        rho_zero,boundary_zero,opt_bound,dec_data=get_fixed_boundary(env_space_df,gamma) 
    else:
        print('not a valid constraint')
    print('zero-cost RR='+str(rho_zero))
    if plot:
        plot_strategy(boundary_zero,r'zero-cost, $\alpha='+str(gamma)+'$',None,dec_data=dec_data)
    
    #compute (reward rate, tdec) joint and posterior average of solution
    reward_rate_dec_time_distr=get_reward_rate_dec_time_distr(boundary_zero,env_space_df,gamma) 
    
    mean_reward=(reward_rate_dec_time_distr.index.get_level_values(1) \
                    *reward_rate_dec_time_distr \
                    /reward_rate_dec_time_distr.sum() \
                     ).sum()
    tmin=reward_rate_dec_time_distr.index.get_level_values(0).min()
    r_max=reward_rate_dec_time_distr.index.get_level_values(1).max()
    r_min=reward_rate_dec_time_distr.index.get_level_values(1).min()
    print(str(gamma)+' '+str(tmin)+' '+str(r_min)+' '+str(mean_reward)+' '+str(r_max))
    
    if knowledge_mode=='full':            
        posterior_expected_reward_rate_sequence=np.asarray([get_MAP_estimate(reward_rate_dec_time_distr,t) for t in range(T+1)])        
        reward_rate_ref=r_max
    elif knowledge_mode=='part': #exponential prior and offset exponential observation model
    
        #these worked well (made plots)
#         if gamma==1/4:
#             baseline=-0.07 #-expected reward_rate
#             slope=0.02
#             cost_rate_seq=baseline+np.asarray([0 if t<tmin else slope*(t-tmin) for t in np.arange(T+1)])
#         else:
#             baseline=-0.07
#             slope=0.03
#             tmin=tmin-2
#             cost_rate_seq=baseline+np.asarray([0 if t<tmin else slope*(t-tmin) for t in np.arange(T+1)])

        posterior_expected_reward_rate_sequence=mean_reward*np.asarray([1 if t<tmin else 1/(1+mean_reward*(t-tmin)) for t in np.arange(T+1)])
        reward_rate_ref=mean_reward
    else:
        print('full or part modes only')

    #get opportunity cost as function of time
    cost_fn=opp_cost
    cost_rate_seq=[reward_rate_to_cost(cost_fn,rr,reward_rate_ref) for rr in posterior_expected_reward_rate_sequence]
    print(cost_rate_seq)
    #solve opportunity cost case
    if bound_constr=='none':
        rho,boundary,dec_data,cost_rate_seq=get_DP_boundary(cost_rate_seq,gamma)
    else:
        tb,b_bel=boundary_zero[1]
        if tb[0]>0:
            b_bel=np.concatenate((np.ones(tb[0]),b_bel))
            tb=np.concatenate((np.arange(tb[0]),tb))
        b_bel=((2*b_bel-1)*np.exp(-cost_sequence)+1)/2
        boundary=get_decb_from_varyb((tb,b_bel))
        rho=get_reward_rate_for_varybound(boundary[1],env_space_df,gamma)
        dec_data=[]
    print('oppor-cost RR='+str(rho))     
    
    if plot:
        plot_strategy(boundary,r'cost, $\alpha='+str(gamma)+'$',cost_rate_seq,dec_data=dec_data)
        fig,ax=pl.subplots()
        ax.plot(cost_sequence,'o-',)
        ax.set_title(r'zero-cost, $\alpha='+str(gamma)+'$')
        
        fig,ax=pl.subplots()
        ax.scatter(reward_rate_dec_time_distr.index.get_level_values(0),reward_rate_dec_time_distr.index.get_level_values(1),s=reward_rate_dec_time_distr.values/10,clip_on=False)
        ax.set_ylim(reward_rate_dec_time_distr.index.get_level_values(1).min(),reward_rate_dec_time_distr.index.get_level_values(1).max())
        ax.set_xlim(0,T)
    
    values=(boundary_zero,reward_rate_dec_time_distr,posterior_expected_reward_rate_sequence,cost_rate_seq,boundary,rho)
    return dict(zip(('b_zero','rr_dt_distr','post_avgrr_seq','c_seq','b_opp','rho_opp'),values))

def solve_DP_cost_strategy(file_path,gamma,plot=False,T=para['T']):
    data_bel_boundary=np.load(file_path)
    tb,b=data_bel_boundary
    tb=np.asarray(tb,dtype=int)
    if data_bel_boundary[0][-1]<T:
        b=np.append(b,np.ones(T-tb[-1]))
        tb=np.append(tb,np.arange(tb[-1]+1,T+1))
        data_bel_boundary=(tb,b)

    data_boundary=get_decb_from_varyb(data_bel_boundary)
    rho,boundary,dec_data,cost_rate_seq=get_DP_boundary(data_boundary,gamma,infer_cost=True) #overload cost_rate_seq iwth data boundary
    
    if plot:
        plot_strategy(boundary,r'cost, $\alpha='+str(gamma)+'$',cost_rate_seq,dec_data=dec_data)
    return rho,cost_rate_seq,data_boundary

#     values=(boundary_zero,reward_rate_dec_time_distr,posterior_expected_reward_rate_sequence,cost_sequence,boundary,rho)
#     return dict(zip(('b_zero','rr_dt_distr','post_avgrr_seq','c_seq','b_opp','rho_opp'),values))
