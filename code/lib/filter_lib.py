import numpy as np
import math
import pandas as pd
from functools import partial
import time
from scipy.optimize import minimize
#with kernprof.py in directory, place @profile above any functions to be profiled , then run:
# kernprof -l tempytron_main.py
#then generate readable output by running:
# python -m line_profiler tempytron_main.py.lprof > profiling_stats.txt 

para=dict()
para['T']=15
para['T_ITI']=7.5 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
para['p']=1/2
para['tp']=0

def get_trial_duration(t,alpha):
    return t+(1-alpha)*(para['T']-t)+para['T_ITI']

def load_data():
    import scipy.io as spio
    mat = spio.loadmat('../data/toktrials.mat', squeeze_me=True)
    mat['toktrials'].flatten().dtype.names
    col_names=mat['toktrials'].flatten().dtype.names
    df_data=pd.DataFrame(columns=col_names)
    for col_name in col_names:
        df_data[col_name]=mat['toktrials'].flatten()[0][col_name]
    df_data.tDecision=(df_data.tDecision-df_data.tFirstTokJump)/200+1 #since first jump at t=1
    #-remove samples with null deicsion times 
    df_data=df_data[~df_data.tDecision.isnull()]
    
    df_data.tDecision=df_data.tDecision.astype('int') 

    #-remove hand designed trials 
    df_data=df_data[~(df_data.sTrialType.isin(df_data.sTrialType.unique()[-2:]) & (df_data.sTrialType.isin(df_data.sTrialType.unique()[-2:])))]
    df_data.reset_index();
    df_data['seq']=df_data.sTokenDirs.apply(lambda x:2*(np.asarray([int(el) for el in list(x)])-1)-1) #note indexing starts at t=1, since len=T not T+1
    #sTrialType :A string describing the trial type: 
    # easy (‘E’), 
    # ambiguous (‘A’), 
    # misleading (‘M’), 
    # bias-for (‘C’ or ‘c’), 
    # random (‘x’), or 
    # unclassified (‘null’)
    df_data['Nt']=df_data.seq.apply(lambda x: np.insert(np.cumsum(x),0,0))
    df_data['nCorrectChoice']=2*(df_data['Nt'].apply(lambda x:x[-1])>0)-1
    df_data.nChoiceMade=2*(df_data.nChoiceMade.values.astype('int')-1)-1 #assuming 1=False, 2=True
    df_data=df_data[['Nt','seq','nPostInterval','idSubject','nCorrectChoice','tDecision','nChoiceMade']]
    from lib.lib import get_pt_plus
    def dummy(Nt_seq):
        return np.array([(get_pt_plus(t,Nt) if Nt>=0 else 1-get_pt_plus(t,Nt)) for t,Nt in enumerate(Nt_seq)])
    df_data['p_suc_seq']=df_data.Nt.apply(lambda x: dummy(x))
    df_data.nPostInterval=df_data.nPostInterval.astype('float64')
    def dummy(row):
        return get_trial_duration(row.tDecision,1-row.nPostInterval/200)
    df_data['duration']=df_data.apply(dummy,axis=1)
    df_data['trialRR']=df_data.apply(lambda row:row.p_suc_seq[int(row.tDecision)],axis=1)/df_data.duration
    return df_data

def filter_step(filtered_value,input_sample,inter_event_interval,filter_factor): #filter factor is 1/(1+filter_timescale)
    tmp=np.power(1-filter_factor,inter_event_interval)
    return tmp*filtered_value+(1-tmp)*input_sample 

def get_contextaware_oppcost_boundary(rho_long,rho_context,T_context,trial_time_vec):
    return rho_long*trial_time_vec+(rho_context-rho_long)*T_context

def get_contextaware_oppcost_boundary(rho_long,trial_time_vec):
    return rho_long*trial_time_vec
#@profile
def get_model_output(trial_seq,model_paras,seed=1):
    '''
    Initializes agent model using given parameter and runs on given trial sequence (State sequence, block index)
    '''
    
    beta_long=1/(1+model_paras['tau_long'])
    #sensitivity_factor=1/(1+((trial_RR-rho_context)) if sample_trial.Index>0 else 1
    #beta_context=1/(1+model_paras['tau_context']*sensitivity_factor)
    #beta_context_plus= 1/(1+model_paras['tau_context_plus']) if model_paras['tau_context_plus'] else model_paras['tau_context']
    #beta_noise=1/(1+model_paras['tau_shared_noise'])

    trial_time_vec=np.arange(para['T']+1)

    #initialize
    dftmp=[]
    #RR_shared_noise=0
    rho_long=0
    rho_context=0
    T_context=0
    #np.random.seed(seed)
    #rng=np.random.default_rng(seed)
    #shared_noise_vec=np.random.normal(0,model_paras['shared_noise_variance_factor'],len(trial_seq))
    #context_noise_vec=np.random.normal(model_paras['context_noise_bias'],model_paras['context_noise_variance_factor'],len(trial_seq))
    for it,sample_trial in enumerate(trial_seq.itertuples()):

        #urgency=(rho_long+model_paras['rho_long_bias'])*trial_time_vec+(rho_context-(rho_long+model_paras['rho_long_bias']))*(T_context+model_paras['Tcontext_bias'])
        
        #boundary=get_contextaware_oppcost_boundary(rho_long,rho_context,T_context,trial_time_vec)
        boundary=get_contextagnostic_oppcost_boundary(rho_long,trial_time_vec)
        #if 0.5>1-urgency[0]:
            #print('too urgent to play')
            #t_decision=1
        #elif len(np.where(sample_trial.p_suc_seq>=1-urgency)[0])==0:
            #print('no length')
        #else:
            #t_decision=int(np.where(sample_trial.p_suc_seq>=1-urgency)[0][0])
        
        try:
            t_decision=int(np.where(sample_trial.p_suc_seq>=1-boundary)[0][0])
        except:
            if 0.5>1-boundary[0]:
                print('too urgent to play')
                t_decision=1
            elif len(np.where(sample_trial.p_suc_seq>=1-boundary)[0])==0:
                #print('no length')
                t_decision=para['T']
            
        Nt_at_tdec=sample_trial.Nt[t_decision]
        nChoiceMade=np.sign(Nt_at_tdec) if np.sign(Nt_at_tdec)!=0 else np.random.choice((-1,1))
        nCorrectChoice=np.sign(sample_trial.Nt[-1])
        duration=get_trial_duration(t_decision,1-sample_trial.nPostInterval/200)

        trial_RR=sample_trial.p_suc_seq[t_decision]/duration
        if it==0:
            rho_long=trial_RR
            rho_context=trial_RR
            T_context=duration
#         RR_private_context_noise=np.random.normal(model_paras['impatience_bias']*trial_RR,model_paras['impatience_bias']*trial_RR)
        #RR_shared_noise = filter_step(RR_shared_noise   , shared_noise_vec[it]*trial_RR, duration, beta_noise)
        #RR_context_noise =context_noise_vec[it]*trial_RR
        #sensitivity_factor=np.power(T_context/duration,sensitivity_pow) if sample_trial.Index>0 else 1

        #switch_cond=trial_RR>rho_long
        #switch_cond=trial_RR<rho_context
        sensitivity_factor=1/(1+np.power(duration/T_context,model_paras['sense_power']))
        beta_context=1/(1+model_paras['tau_context']*sensitivity_factor)

        rho_long =   filter_step(rho_long   , trial_RR, duration, beta_long)
        rho_context =filter_step(rho_context, trial_RR, duration, beta_context)# if switch_cond else beta_context_plus)
        T_context=   filter_step(T_context  , duration, duration, beta_context)# if switch_cond else beta_context_plus)
        #rho_long =   filter_step(rho_long   , trial_RR + RR_shared_noise, duration, beta_long)
        #rho_context =filter_step(rho_context, trial_RR + RR_shared_noise + RR_context_noise, duration, beta_context if trial_RR<rho_context else beta_context_plus)
        #rho_context =filter_step(rho_context, trial_RR + RR_context_noise, duration, beta_context if trial_RR<rho_context else beta_context_plus)
       #T_context=   filter_step(T_context  , duration , duration, beta_context if trial_RR<rho_context else beta_context_plus)

        dftmp.append({'seq':sample_trial.seq, 
                    'nChoiceMade':nChoiceMade,
                    'nCorrectChoice':nCorrectChoice,
                    'tDecision':t_decision,
                    'nPostInterval':sample_trial.nPostInterval,
                    'trialRR':trial_RR,
                    'rho_long':rho_long,
                    'rho_context':rho_context,
                    'T_context':T_context,
                    'duration':duration,
                    'sensitivity_factor':sensitivity_factor
                     })
    return pd.DataFrame(dftmp)

def get_transition_ensemble(df_tmp,measure='tDecision'):
    
    time_depth=100
    history=20
    block_len_range=[90,110]
    block_len_range=[-1,np.Inf]
    timevec=np.arange(-history,time_depth)
    
    #group transitions by fast-to-slow and slow-to-fast 
    df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff(periods=1) #dx_i=x_i-x_i-1
    df_tmp.loc[0,'nPostInterval_diff']=0 #replace nan ###remove initial block (an outlier in length)
    cond=df_tmp['nPostInterval_diff']!=0 #location of trial type switch,i.e. block start
    df_tmp['block_len']=0
    df_tmp.loc[cond,'block_len']=np.diff(np.insert(df_tmp[cond].index.values,cond.sum(),len(df_tmp))) #tag block starts with corresponding block length.np.diff uses period=-1 so add last block.
    cond=cond & (df_tmp['block_len']>=block_len_range[0]) & (df_tmp['block_len']<=block_len_range[1])#lower bound on block size
    
    data_store=[]
    block_times=[150,50]
    for b in block_times:
        num_samples=(cond & (df_tmp.nPostInterval==b)).sum()
        data=np.zeros((num_samples,len(timevec)))
        for it,ind in enumerate(df_tmp[cond & (df_tmp.nPostInterval==b)].index.values):
            data[it,:]=df_tmp.iloc[ind-history:ind+time_depth][measure].values
        data_store.append(data)
    
    return timevec+1,data_store #index with trials after switch
    
def error_fn(data_actual,data_model,mode=0):
    if mode==0:
        return np.mean(np.power(data_actual-data_model,2))
   
def get_error(paras,df_act,data_store_actual,model='taus_only'):
    if model=='taus_only':
        paras={'tau_context':paras[0], 
                #'tau_context_plus':paras[0],#paras[2], 
                'sense_power':paras[2], 
                'tau_long':paras[1],
                'Tcontext_bias':0,#paras[1],
                'rho_long_bias':0,#paras[3],
                'shared_noise_variance_factor':0,
                'tau_shared_noise':10,
                'context_noise_bias':0,
                'context_noise_variance_factor':0
                }
    df_mod=get_model_output(df_act,paras)
    timevec,data_store_model=get_transition_ensemble(df_mod)
    #return [error_fn(d_a,d_m) for d_a,d_m in zip([data_store_actual[1]],[data_store_model[1]])][0]
    return sum([error_fn(d_a,d_m) for d_a,d_m in zip(data_store_actual,data_store_model)])/2 #equal 1/2 weighting fast to slow and slow to fast
    #return sum([error_fn(np.concatenate([d_a[:,:20],d_a[:,-20:]]),np.concatenate([d_m[:,:20],d_m[:,-20:]])) for d_a,d_m in zip(data_store_actual,data_store_model)])/2 #equal 1/2 weighting fast to slow and slow to fast

def run_para_grid(func,paravecs):
    dims=[len(pvec) for pvec in paravecs]
    func_vals=np.zeros(dims)
    it_progress=0
    st=time.time()
    for it in range(np.prod(dims)):
        inds=np.unravel_index(it,dims)
        paras=[pvec[ind] for pvec,ind in zip(paravecs,inds)]
        if paras[0]<=paras[1]:# and paras[2]<paras[1]: #tau long (paravecs[1]) should be larger than context versions
            func_vals[inds]=func(paras)
        if it%np.prod(dims[1:])==0:
            it_progress+=1
            print(str(it_progress) +' of '+str(dims[0])+ ' took '+str(time.time()-st))
            st=time.time()
    return func_vals

def callback(paras,nparas): 
    '''prints iteration info. called by scipy.minimize'''
    global curr_iter
    print(''.join(['{0:d} ']+['{'+str(it)+':3.6f} ' for it in range(1,len(paras)+1)]).format(*([curr_iter]+list(paras))))
    curr_iter += 1
    
def run_para_opt(func,init_paras):
    nullfunctol=1e-4
    nullmaxiter=200
    header=['Iter','tauc','Tbias','taucp']#,'rhobias']
    print(''.join(['{'+str(it)+':9s} ' for it in range(len(init_paras)+1)]).format(*header))
    global curr_iter
    curr_iter = 1
    callbackp=partial(callback,nparas=len(init_paras))
    #method='SLSQP'
    method='Nelder-Mead'
    outstruct = minimize(func, init_paras, method=method, callback=callbackp, options={'fatol':nullfunctol ,'xatol':nullfunctol ,'disp': True,'maxiter':nullmaxiter})
    return outstruct
      
def run_evaluation(mode,subject,df_data,root_filename,model_type=None):
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)
    timevec,data_store_actual=get_transition_ensemble(df_act)
                         
    #select objective
    part_error_fn=partial(get_error,df_act=df_act,data_store_actual=data_store_actual,model='taus_only')
    
    #run
    average_trial_duration=get_trial_duration(6.5,0.5)
    step_size=average_trial_duration
    if mode=='grid':
        paravecs=[np.power(2,np.linspace(0,10,51))*step_size]*3
        #paravecs=[paravecs[0],np.arange(-4,5),paravecs[2]]
        st=time.time()
        paravecs=[paravecs[0],np.power(2,np.linspace(0,10,51))*step_size]
        error_grid=run_para_grid(part_error_fn,paravecs)
        print('grid took '+str(time.time()-st))
        np.save('error_grid_'+root_filename+'.npy',error_grid)
        np.save('error_grid_'+root_filename+'_pvecs.npy',paravecs)
    elif mode=='opt':
        
        error_grid=np.load('error_grid_v9_grid_neldermead_finejusttaus_1_1000.npy')
        error_grid[error_grid==0]=np.nan
        tau_shortvec=np.power(2,np.linspace(0,10,51))*average_trial_duration
        tau_longvec=np.power(2,np.linspace(0,10,51))*average_trial_duration
        optinds=np.unravel_index(np.nanargmin(error_grid),(len(tau_shortvec),len(tau_longvec)))
        para_init=np.array([tau_shortvec[optinds[0]],tau_longvec[optinds[1]]])
        
        #para_init=np.asarray([8*step_size,5,50*step_size])  
        #para_init=np.array([ (3.66738620e+02)/4, -5*(-7.12962708e-01),  7.88915814e+02,  -0.003])
        para_init=np.array([600,20000,15])

        st=time.time()
        outstruct=run_para_opt(part_error_fn,para_init)
        print('opt took '+str(time.time()-st))
        opt_paras=outstruct.x
        print(opt_paras)
        np.save('opt_paras_'+root_filename+'.npy',opt_paras)
    else:
        print('mode is opt or grid')
def check_error(subject,df_data,paras):
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)
    timevec,data_store_actual=get_transition_ensemble(df_act)
                         
    #select objective
    part_error_fn=partial(get_error,df_act=df_act,data_store_actual=data_store_actual,model='taus_only')
    
    return part_error_fn(paras)