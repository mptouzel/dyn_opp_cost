import numpy as np
import math
import pandas as pd
from functools import partial
import time
from scipy.optimize import minimize
from multiprocessing import Pool
#with kernprof.py in directory, place @profile above any functions to be profiled , then run:
# kernprof -l tempytron_main.py
#then generate readable output by running:
# python -m line_profiler tempytron_main.py.lprof > profiling_stats.txt 
from lib.lib import get_trial_duration
para=dict()
if True:
    para['T']=15
    para['T_ITI']=7.5 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
    para['p']=1/2
    para['tp']=0
else:
    para['T']=11
    para['T_ITI']=0 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
    para['p']=1/2
    para['tp']=0
    
def get_block_ids(df_data):
    df_data['block_idx']=0
    for subject_id in [1,2]:
        df_tmp=df_data.loc[df_data.idSubject==subject_id,:]
        df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff(periods=1) #dx_i=x_i-x_i-1
        df_tmp.loc[0,'nPostInterval_diff']=0 #replace nan ###remove initial block (an outlier in length)
        cond_type=df_tmp['nPostInterval_diff']!=0 #location of trial type switch,i.e. block start

        df_tmp['date_diff']=df_tmp.dDate.dt.day.diff(periods=1)
        df_tmp.loc[0,'date_diff']=0
        cond_date=(df_tmp['date_diff']!=0) #location of day increment

        cond=cond_type | cond_date #at which either context or day changes (the latter correctly splits two contiguous blocks of the same type if they happen on successive days
        df_data.loc[df_data.idSubject==subject_id,'block_idx']=cond.cumsum()
    return df_data
    
def load_data():
    import scipy.io as spio
    mat = spio.loadmat('../../exp_data/Thura_etal_2016/toktrials.mat', squeeze_me=True)
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
    df_data['dDate']=pd.to_datetime(df_data['dDate'])
    df_data=get_block_ids(df_data)
    df_data=df_data[['Nt','seq','nPostInterval','idSubject','nCorrectChoice','tDecision','nChoiceMade','dDate','block_idx']]
    df_data=postprocess_df(df_data)
    return df_data

def postprocess_df(df_data):
    from lib.lib import get_pt_plus
    def dummy(Nt_seq):
        return np.array([(get_pt_plus(t,Nt) if Nt>=0 else 1-get_pt_plus(t,Nt)) for t,Nt in enumerate(Nt_seq)])
    df_data['p_plus']=df_data.Nt.apply(lambda x: np.array([get_pt_plus(t,Nt) for t,Nt in enumerate(x)]))
    df_data['p_success']=df_data.p_plus.apply(lambda x: np.asarray([max([xel,1-xel]) for xel in x]))
    df_data.nPostInterval=df_data.nPostInterval.astype('float64')
    def dummy(row):
        return get_trial_duration(row.tDecision,1-row.nPostInterval/200)
    df_data['duration']=df_data.apply(dummy,axis=1)
    df_data['trialRR']=df_data.apply(lambda row:row.p_success[int(row.tDecision)],axis=1)/df_data.duration
    return df_data

def filter_step(filtered_value,input_sample,inter_event_interval,filter_factor): #filter factor is 1/(1+filter_timescale)
    tmp=np.power(1-filter_factor,inter_event_interval)
    return tmp*filtered_value+(1-tmp)*input_sample 

def get_contextaware_oppcost_boundary(rho_long,rho_context,T_context,trial_time_vec):
    return rho_long*trial_time_vec+(rho_context-rho_long)*T_context

def get_contextagnostic_oppcost_boundary(rho_long,trial_time_vec):
    return rho_long*trial_time_vec
def get_optimal_foraging_boundary(rho_long,trial_time_vec,postint):
    return 1+rho_long*trial_time_vec#get_trial_duration(trial_time_vec,1-postint/200)

#@profile
def get_model_output(trial_seq,model_paras):
    '''
    Initializes agent model using given parameter and runs on given trial sequence (State sequence, block index)
    '''
    
    opp_cost_mode='trial_aware'
    
    beta_long=1/(1+model_paras['tau_long'])
    k_unitconv=model_paras['unitconv']

    trial_time_vec=np.arange(para['T']+1)

    #initialize
    dftmp=[]
    rho_long=0
    rho_context=0
    T_context=0
    T_context_long=0
    duration=0
    for it,sample_trial in enumerate(trial_seq.itertuples()):
        
        
        if opp_cost_mode=='trial_aware':
            opp_cost=get_contextaware_oppcost_boundary(rho_long,rho_context,duration,trial_time_vec)
        else:
            opp_cost=get_contextagnostic_oppcost_boundary(rho_long,trial_time_vec)

        asym_flag=False
        if asym_flag:
            b=model_paras['b']
            asym_p=np.ones(len(sample_trial.Nt))#sample_trial.Nt>=0
            asym_m=np.ones(len(sample_trial.Nt))#sample_trial.Nt<=0
            min_regret=np.min(np.vstack(((1+b*asym_p)-(sample_trial.p_plus)*(1+b*asym_p),(1-b*asym_m)-(1-sample_trial.p_plus)*(1-b*asym_m))),axis=0) #action based
        else:
            min_regret=1-sample_trial.p_success #action based
            
        if len(np.where(min_regret<=opp_cost)[0])>0:
            t_decision=int(np.where(min_regret<=opp_cost)[0][0]) #need to redefine this via p^+
        else:
            t_decision=para['T']
        t_decision=t_decision if t_decision<=para['T'] else para['T']
        duration=get_trial_duration(t_decision,1-sample_trial.nPostInterval/200)

        Nt_at_tdec=sample_trial.Nt[t_decision]
        if asym_flag:
            nChoiceMade=np.random.choice((-1,1)) if (sample_trial.p_plus[t_decision])*(1+b*asym_p[t_decision])==(1-sample_trial.p_plus[t_decision])*(1-b*asym_p[t_decision]) else (-1)**np.argmax(((sample_trial.p_plus[t_decision])*(1+b*asym_m[t_decision]),(1-sample_trial.p_plus[t_decision])*(1-b*asym_m[t_decision]))) 
            trial_RR=np.max(((sample_trial.p_plus[t_decision])*(1+b*asym_p[t_decision]),(1-sample_trial.p_plus[t_decision])*(1-b*asym_m[t_decision])))/duration
        else:
            nChoiceMade=np.sign(Nt_at_tdec) if np.sign(Nt_at_tdec)!=0 else np.random.choice((-1,1),p=[1/2,1/2])
            trial_RR=k_unitconv*sample_trial.p_success[t_decision]/duration

        nCorrectChoice=np.sign(sample_trial.Nt[-1])

        if it==0:
            T_context=duration
            rho_long=trial_RR
            rho_context=trial_RR
        
        rho_long =   filter_step(rho_long   , trial_RR, duration, beta_long)

        if opp_cost_mode=='trial_aware':
            sensitivity_factor=1 if len(model_paras)==3 else 1/(1+np.power(duration/T_context,model_paras['sense_power']))

            beta_context=1/(1+model_paras['tau_context']*sensitivity_factor)

            rho_context = filter_step(rho_context, trial_RR, duration, beta_context)# if switch_cond else beta_context_plus)
            T_context=    filter_step(T_context  , duration, duration, beta_context)# if trial_RR<rho_context else beta_context_plus)
        
        dftmp.append({'seq':sample_trial.seq,
                    'nChoiceMade':nChoiceMade,
                    'nCorrectChoice':nCorrectChoice,
                    'tDecision':t_decision,
                    'nPostInterval':sample_trial.nPostInterval,
                    'trialRR':trial_RR,
                    'rho_long':rho_long,
                    'rho_context':rho_context if opp_cost_mode=='trial_aware' else np.nan,
                    'T_context':T_context if opp_cost_mode=='trial_aware' else np.nan,
                    'sensitivity_factor':sensitivity_factor if opp_cost_mode=='trial_aware' else np.nan,
                    'duration':duration,
                    'block_idx':sample_trial.block_idx,
                    'dDate':sample_trial.dDate
                     })
    return pd.DataFrame(dftmp)

def get_transition_ensemble(df_tmp,measure='tDecision'):
    
    time_depth=100
    history=20
    timevec=np.arange(-history,time_depth)

    #df_tmp['block_day']=(df_tmp.groupby('block_idx').first().dDate-df_tmp.groupby('block_idx').first().dDate[0]).dt.days
    df_tmp['block_day']=(df_tmp.dDate-df_tmp.dDate[0]).dt.days

    data_store=[]
    block_times=[150,50]
    for b in block_times:
        block_lens=df_tmp[df_tmp.nPostInterval==b].block_idx.value_counts().sort_index().values
        data=np.empty((((df_tmp.nPostInterval==b)>0).sum(),len(timevec)))
        data[:]=np.nan
        
        startinds=df_tmp[df_tmp.nPostInterval==b].reset_index().groupby('block_idx').first()['index'].values[:-1]
        it=-1
        for idx,start_idx in enumerate(startinds):
            if idx>0:
                if df_tmp.iloc[start_idx-1].block_day==df_tmp.iloc[start_idx].block_day:
                    duration=min([block_lens[idx],time_depth])
                    it+=1
                    data[it,:history+duration]=df_tmp.iloc[start_idx-history:start_idx+duration][measure].values
        data_store.append(data[:it+1])
    
    return timevec+1,data_store #index with trials after switch
    
def error_fn(data_actual,data_model):
    target=np.power(np.nanmean(data_actual,axis=0)-np.nanmean(data_model,axis=0),2)
    #target+=np.power(np.std(data_actual,axis=0)-np.std(data_model,axis=0),2)
    return np.nanmean(target)
       
def get_error_for_grid0(paras,df_act,data_store_actual,opt_paras,model='default'):
    return get_error(np.asarray([paras[0],paras[1],opt_paras[2],opt_paras[3]]),df_act,data_store_actual,model='default')
def get_error_for_grid1(paras,df_act,data_store_actual,opt_paras,model='default'):
    return get_error(np.asarray([paras[0],opt_paras[1],paras[1],opt_paras[3]]),df_act,data_store_actual,model='default')
def get_error_for_grid2(paras,df_act,data_store_actual,opt_paras,model='default'):
    return get_error(np.asarray([paras[0],opt_paras[1],opt_paras[2],paras[1]]),df_act,data_store_actual,model='default')

def get_error(paras,df_act,data_store_actual,model='default'):
    if model=='default':
        paras={'tau_context':paras[0], 
               'tau_long':paras[1],
               'unitconv': paras[2],
               'sense_power':paras[3]
                }
    elif model=='slow_to_fast_only':
        paras={'tau_long':paras[0], #100*paras[0]#,
               'unitconv': paras[1]
               #'tau_context':paras[0], 
               #'tau_long':paras[1], #100*paras[0]#,
               #'unitconv': paras[2]
                }
    else:
        print(model+' not found')
    df_mod=get_model_output(df_act,paras)

    timevec,data_store_model=get_transition_ensemble(df_mod)
    if model=='default':
        return sum([error_fn(d_a,d_m) for d_a,d_m in zip(data_store_actual,data_store_model)])/2 #equal 1/2 weighting fast to slow and slow to fast
    elif model=='slow_to_fast_only':
        e_out=[error_fn(d_a,d_m) for d_a,d_m in zip([data_store_actual[1]],[data_store_model[1]])][0] 
        return e_out
    else:
        print(model+' not found')
    #return sum([error_fn(np.concatenate([d_a[:,:20],d_a[:,-20:]]),np.concatenate([d_m[:,:20],d_m[:,-20:]])) for d_a,d_m in zip(data_store_actual,data_store_model)])/2 #equal 1/2 weighting fast to slow and slow to fast

def run_para_halfgrid(func,paravecs):
    dims=[len(pvec) for pvec in paravecs]
    func_vals=np.zeros(dims)
    it_progress=0
    st=time.time()
    for it in range(np.prod(dims)):
        inds=np.unravel_index(it,dims)
        paras=[pvec[ind] for pvec,ind in zip(paravecs,inds)]
        if paras[0]<=paras[1]:
            func_vals[inds]=func(paras)
        if it%np.prod(dims[1:])==0:
            it_progress+=1
            print(str(it_progress) +' of '+str(dims[0])+ ' took '+str(time.time()-st))
            st=time.time()
    return func_vals

def run_para_grid(func,paravecs):
    dims=[len(pvec) for pvec in paravecs]
    func_vals=np.zeros(dims)
    it_progress=0
    st=time.time()
    for it in range(np.prod(dims)):
        inds=np.unravel_index(it,dims)
        paras=[pvec[ind] for pvec,ind in zip(paravecs,inds)]
        func_vals[inds]=func(paras)
        if it%np.prod(dims[1:])==0:
            it_progress+=1
            print(str(it_progress) +' of '+str(dims[0])+ ' took '+str(time.time()-st))
            st=time.time()
    return func_vals

def run_para_grid_par(func,paravecs):
    
    run_para_row_part=partial(run_para_row,func=func,paravecs=paravecs)
    print('start pool')
    pool = Pool(processes=2)              # process per core
    print('run pool')
    pool.map(run_para_row_part, np.arange(len(paravecs[0])))
    pool.close()
    pool.join()
    print('end pool')
    func_vals=np.zeros((len(paravecs[0]),len(paravecs[1])))
    for it in range(len(paravecs[0])):
        func_vals[it,:]=np.load('funcvals_'+str(it)+'.npy')
    return func_vals

def run_para_row(it,func,paravecs):
    func_vals=np.zeros(len(paravecs[0]))
    for pit,para in enumerate(paravecs[1]):
        if paravecs[0][it]<=para:
            print(pit)
            func_vals[pit]=func([paravecs[0][it],para])
    np.save('funcvals_'+str(it)+'.npy',func_vals)

def callback(paras,nparas): 
    '''prints iteration info. called by scipy.minimize'''
    global curr_iter
    print(''.join(['{0:d} ']+['{'+str(it)+':3.6f} ' for it in range(1,len(paras)+1)]).format(*([curr_iter]+list(paras))))
    curr_iter += 1
    
def run_para_opt(func,init_paras):
    nullfunctol=1e-4
    nullmaxiter=200
    if len(init_paras)==3:
        header=['Iter','tauc']#,'taul','unitconv']
    else:
        header=['Iter','tauc','taul','unitconv','sensepow']
    header=['Iter','taul','unitconv']
    print(''.join(['{'+str(it)+':9s} ' for it in range(len(init_paras)+1)]).format(*header))
    global curr_iter
    curr_iter = 1
    callbackp=partial(callback,nparas=len(init_paras))
    #method='SLSQP'
    method='Nelder-Mead'
    outstruct = minimize(func, init_paras, method=method, callback=callbackp, options={'fatol':nullfunctol ,'xatol':nullfunctol ,'disp': True,'maxiter':nullmaxiter})
    return outstruct
      
def run_evaluation(mode,subject,df_data,root_dir,filename,model_type=None,plane_idx=0):
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)
    #df_tmp=df_act
    #df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff(periods=1) #dx_i=x_i-x_i-1
    #df_tmp.loc[0,'nPostInterval_diff']=0 #replace nan ###remove initial block (an outlier in length)
    #cond=df_tmp['nPostInterval_diff']!=0 #location of trial type switch,i.e. block start
    #df_tmp['block_len']=0
    #block_len_range=[-1,np.Inf]
    #df_tmp.loc[cond,'block_len']=np.diff(np.insert(df_tmp[cond].index.values,cond.sum(),len(df_tmp))) #tag block starts with corresponding block length.np.diff uses period=-1 so add last block.
    #df_act=df_tmp.drop(df_tmp[cond].index.values[250:360]).reset_index(drop=True)
    
    timevec,data_store_actual=get_transition_ensemble(df_act)
                         
    #select objective
    opt_paras=np.asarray([188.104904, 56605.906820, 1.151912, 2.308570]) if subject==2 else np.asarray([646.565430, 45248.788045, 1.038641, 11.703220])
    #run
    average_trial_duration=get_trial_duration(6.5,0.5)
    step_size=average_trial_duration
    if mode=='grid':
        #opt_paras=np.load(root_dir+'opt_paras_'+filename+'.npy')
        
        #(tau_long,tau_context)
        if plane_idx==0:
            paravecs=[np.logspace(1,4,30),np.logspace(1,5.5,40)]
            part_error_fn=partial(get_error_for_grid0,opt_paras=opt_paras,df_act=df_act,data_store_actual=data_store_actual,model=model_type)
        elif plane_idx==1:
            paravecs=[np.logspace(1,4,20),np.linspace(0.9,1.3,20)]
            part_error_fn=partial(get_error_for_grid1,opt_paras=opt_paras,df_act=df_act,data_store_actual=data_store_actual,model=model_type)
        elif plane_idx==2:
            paravecs=[np.logspace(1,4,20),np.linspace(-5,20,20)]
            part_error_fn=partial(get_error_for_grid2,opt_paras=opt_paras,df_act=df_act,data_store_actual=data_store_actual,model=model_type)
        
        #paravecs=[paravecs[0],paravecs[1].opt_paras[2],opt_paras[3]]
        st=time.time()
        if plane_idx==0:
            error_grid=run_para_halfgrid(part_error_fn,paravecs)
        else:
            error_grid=run_para_grid(part_error_fn,paravecs)
        print('grid took '+str(time.time()-st))
        np.save(root_dir+'error_grid_'+filename+'.npy',error_grid)
        np.save(root_dir+'error_grid_'+filename+'_pvecs.npy',paravecs)
        
        ##(nu,tau_context)
        #paravecs=[np.power(2,np.linspace(0,10,51))*step_size]*2
        #paravecs=[paravecs[0],paravecs[1].opt_paras[2],opt_paras[3]]
        #st=time.time()
        #error_grid=run_para_grid(part_error_fn,paravecs)
        #print('grid took '+str(time.time()-st))
        #np.save(root_dir+'error_grid_'+filename+'.npy',error_grid)
        #np.save(root_dir+'error_grid_'+filename+'_pvecs.npy',paravecs)
        
    elif mode=='opt':
        
        #error_grid=np.load(root_dir+'error_grid_v9_grid_neldermead_finejusttaus_1_1000.npy')
        #error_grid[error_grid==0]=np.nan
        #tau_shortvec=np.power(2,np.linspace(0,10,51))*average_trial_duration
        #tau_longvec=np.power(2,np.linspace(0,10,51))*average_trial_duration
        #optinds=np.unravel_index(np.nanargmin(error_grid),(len(tau_shortvec),len(tau_longvec)))
        #para_init=np.array([tau_shortvec[optinds[0]],tau_longvec[optinds[1]]])
        part_error_fn=partial(get_error,df_act=df_act,data_store_actual=data_store_actual,model=model_type)

        if model_type=='default':
            if subject==1:
                para_init=np.array([694.464372 ,38160.717270 ,1.0386 ,11.703220])
            else:
                para_init=np.array([1.42554012e+02, 4.30979938e+04,1.15,8.19938272e+00])

        elif model_type=='slow_to_fast_only':
            para_init=np.array([300,10000,1])
            para_init=np.array([100,1])
        else:
            print(model+' not found')
            
        st=time.time()
        outstruct=run_para_opt(part_error_fn,para_init)
        print('opt took '+str(time.time()-st))
        opt_paras=outstruct.x
        print(opt_paras)
        print('saved in '+root_dir+'opt_paras_'+filename+'.npy')
        np.save(root_dir+'opt_paras_'+filename+'.npy',opt_paras)
    else:
        print('mode is opt or grid')
        
def check_error(subject,df_data,paras,model_type='default'):
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)
    timevec,data_store_actual=get_transition_ensemble(df_act)
                         
    #select objective
    part_error_fn=partial(get_error,df_act=df_act,data_store_actual=data_store_actual,model=model_type)
    
    return part_error_fn(paras)