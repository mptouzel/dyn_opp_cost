# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#load libraries and setup file
import sys,os
root_path = os.path.abspath(os.path.join('..'))
print(root_path)
if root_path not in sys.path:
    sys.path.append(root_path)
# %run -i "../lib/utils/ipynb_setup.py"
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

#plotting
import matplotlib.pyplot as pl
import seaborn as sns
sns.set_style("ticks", {'axes.grid' : True})
pl.rc("figure", facecolor="white",figsize = (8,8))
#pl.rc("figure", facecolor="gray",figsize = (8,8))
pl.rc('text', usetex=True)
pl.rc('text.latex', preamble=[r'\usepackage{amsmath}'])
pl.rc('lines',markeredgewidth = 2)
pl.rc('font',size = 12)

fig_dir='../../output/figures/'
data_dir='../../output/data/'

#check these are consistent with dyn_prog_model.npy
para=dict()
para['T']=15
para['T_ITI']=7.5 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
para['p']=1/2
para['tp']=0
blockname=('slow block','fast block')
block_times=[150,50]
from lib.filter_lib import *
from lib.filter_plotting import *

from lib.dyn_prog_model import get_pt_plus
from lib.lib import get_trajs
from seaborn import color_palette

df_traj=get_trajs() #loads all possible token trajectories for default settings

# ## Periodic alpha dynamics (fig.2 & supp. fig. 2)

# Generate simulated trial sequence

# +
num_blocks=500
block_size=300
alpha_sequence=np.zeros(num_blocks)
alpha_sequence[1::2]=1/4
alpha_sequence[::2]=3/4
alpha_sequence=np.repeat(alpha_sequence,[block_size]*num_blocks)
num_trials=len(alpha_sequence)

np.random.seed(0)
df_sim=df_traj.sample(num_trials,replace=True).reset_index(drop=True)
df_sim['nPostInterval']=(1-alpha_sequence)*200
df_sim['block_idx']=np.repeat(np.arange(num_blocks),[block_size]*num_blocks)
df_sim['dDate']=np.nan
# -

model_paras_S1_taus={'tau_context':20*10, #about trial duration x10
                'sense_power':9,        #approximately the value learned from subject 1
                'tau_long':block_size*20*10,     #
                'unitconv':1
                }
df_mod=get_model_output(df_sim,model_paras_S1_taus)
df_mod['Nt']=df_sim.Nt
df_mod['p_plus']=df_sim.p_plus
df_mod['p_success']=df_sim.p_success

T_avg=20
block_size=300
T_exp=1e8
num_blocks=int(T_exp/T_avg/block_size)
max_trials=int(5e6)#num_blocks*block_size
df_sim=df_traj.sample(max_trials,replace=True).reset_index(drop=True)

# +
error_store=[]
times_store=[]

alpha_sequence=np.zeros(num_blocks)
alpha_sequence[1::2]=1/4
alpha_sequence[::2]=3/4
alpha_sequence=np.repeat(alpha_sequence,[block_size]*num_blocks)
df_tmp=df_sim.iloc[:max_trials]
df_tmp['nPostInterval']=(1-alpha_sequence)*200
tau_long_vec=[1e3,1e4,1e5]
for tau_long in tau_long_vec:
    st=time.time()    

    model_paras_S1_taus={'tau_context':407.047901,#/average_trial_duration,
                    'sense_power':8.839935, 
                    'tau_long':tau_long,#31620.805644,
                    'Tcontext_bias':0,
                    'shared_noise_variance_factor':0,
                    'tau_shared_noise':30,
                    'context_noise_bias':0,
                    'context_noise_variance_factor':0
                    }
    df_mod=get_model_output(df_tmp,model_paras_S1_taus)
    rho_act=(df_mod.nChoiceMade==df_mod.nCorrectChoice).sum()/df_mod.duration.sum()

    error=np.cumsum(np.fabs(df_mod.rho_long.values-rho_act)*df_mod.duration)/df_mod.duration.cumsum()/rho_act
    step=100
    error_store.append(error[::step])
    times_store.append(df_mod.duration.cumsum()[::step])
    print(time.time()-st)
np.save(data_dir+'error_store.npy',error_store)
np.save(data_dir+'times_store.npy',times_store)

# +
T_avg=20
block_size=300
T_exp=1e8

block_size_vec=[100,200,400]

num_blocks=int(T_exp/T_avg/block_size_vec[-1])
max_trials=num_blocks*block_size_vec[-1]

tau_long_vec=[1e3,1e4,1e5]

error_store2=np.zeros((len(tau_long_vec),len(block_size_vec)))
for tit,tau_long in enumerate(tau_long_vec):
    for bit,block_size in enumerate(block_size_vec):
        st=time.time() 
        num_blocks=int(max_trials/block_size)
        max_trials=num_blocks*block_size
        alpha_sequence=np.zeros(num_blocks)
        alpha_sequence[1::2]=1/4
        alpha_sequence[::2]=3/4
        alpha_sequence=np.repeat(alpha_sequence,[block_size]*num_blocks)
        df_tmp=df_sim.iloc[:max_trials]
        df_tmp['nPostInterval']=(1-alpha_sequence)*200
        model_paras_S1_taus={'tau_context':407.047901,#/average_trial_duration,
                        'sense_power':8.839935, 
                        'tau_long':tau_long,#31620.805644,
                        'Tcontext_bias':0,
                        'shared_noise_variance_factor':0,
                        'tau_shared_noise':30,
                        'context_noise_bias':0,
                        'context_noise_variance_factor':0
                        }
        df_mod=get_model_output(df_tmp,model_paras_S1_taus)
        rho_act=(df_mod.nChoiceMade==df_mod.nCorrectChoice).sum()/df_mod.duration.sum()
        error_store2[tit,bit]=np.sum(np.fabs(df_mod.rho_long.values-rho_act)*df_mod.duration)/df_mod.duration.sum()/rho_act
        print(time.time()-st)
np.save(data_dir+'errorstore2.npy',error_store2)

# +
df_tmp=df_mod

colors=color_palette("colorblind", 10)
colreg=colors[0]
collong=colors[4]
colcon=colors[2]

df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff(periods=1) #dx_i=x_i-x_i-1
df_tmp.loc[0,'nPostInterval_diff']=0 #replace nan ###remove initial block (an outlier in length)
cond=df_tmp['nPostInterval_diff']!=0 #location of trial type switch,i.e. block start
df_tmp['block_len']=0
block_len_range=[-1,np.Inf]
df_tmp.loc[cond,'block_len']=np.diff(np.insert(df_tmp[cond].index.values,cond.sum(),len(df_tmp))) #tag block starts with corresponding block length.np.diff uses period=-1 so add last block.
nblocks=(df_tmp.block_len>0).sum()

block_choice=int(nblocks/2)+2
block_ind=df_tmp.loc[cond].index.values[block_choice]
margin_size=20
block_range=(block_ind-margin_size,block_ind+df_tmp.loc[block_ind].block_len+margin_size)
traj_it=block_ind+int(df_tmp.loc[block_ind].block_len/2) #set as half way through block
alp=1-df_tmp.loc[traj_it].nPostInterval/200
Nt_traj=df_tmp.loc[traj_it].Nt

Tdec=df_tmp.loc[traj_it].tDecision

fig,ax=pl.subplots(3,3,figsize=(8,8))

tlim=(2,12)

#experiment
axind=(2,0)
ax[axind].plot(df_tmp[cond].block_len.values,'-',color=[0.7]*3)
tmp=df_tmp[cond].reset_index(drop=True)
for it in range(2):
    ax[axind].plot(tmp[tmp.nPostInterval==block_times[it]].block_len+2*(-1)**it,'.',color=colors[it],label=blockname[it],ms=0.3)
ax[axind].plot([block_choice],[df_tmp.loc[block_ind].block_len],'o',mec='k',mfc=colors[1])
ax[axind].text(0.1,0.9,'experiment',transform=ax[axind].transAxes)
ax[axind].set_ylim(0,2*block_size)#2000)
ax[axind].set_ylabel('block length')
ax[axind].set_xlabel('block index')
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].spines['left'].set_visible(False)

#block
axind=(1,0)
ax[axind].text(0.1,0.9,'block',transform=ax[axind].transAxes)
ax[axind].plot(1-df_tmp.iloc[block_range[0]:block_range[1]].nPostInterval/200,'k--')
alpvec=1-df_tmp.iloc[block_range[0]:block_range[1]].nPostInterval/200
ax[axind].plot(alpvec[alpvec==1/4],'.',color=colors[0])
ax[axind].plot(alpvec[alpvec==3/4],'.',color=colors[1])
ax[axind].plot([traj_it],[1-df_tmp.iloc[traj_it].nPostInterval/200],'o',mec='k',mfc='None')
ax[axind].set_ylim(0,1)
ax[axind].set_yticks([0,0.25,0.75,1])
ax[axind].set_yticklabels([r'$0$',r'$1/4$',r'$3/4$',r'$1$'])
ax[axind].set_ylabel(r'incentive strength, $\alpha$')
ax[axind].set_xlabel('trial index')
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].spines['left'].set_visible(False)
ax[axind].set_xlim(block_range)

#evidence space
axind=(0,0)
ax[axind].text(0.1,0.9,'trial',transform=ax[axind].transAxes)
for Nm in np.arange(Tdec):
    ax[axind].plot(np.arange(Tdec+1-Nm)+Nm,np.arange(Tdec+1-Nm)-Nm,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
ax[axind].plot([Tdec],[-Tdec],'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
ax[axind].fill_between(np.arange(para['T']+1),-np.arange(para['T']+1),np.arange(para['T']+1),color=[0.9]*3)
ax[axind].plot(np.arange(Tdec,para['T']+1),np.arange(Tdec,para['T']+1),'k--')
ax[axind].plot([Tdec]*2,[Tdec,para['T']],'k--')
arrow_start=(para['T'],para['T'])
arrow_end=(Tdec+(1-alp)*(para['T']-Tdec),para['T'])
ax[axind].annotate("",xy=arrow_end,xycoords='data',
               xytext=arrow_start,textcoords='data',
               arrowprops=dict(arrowstyle="->",connectionstyle='arc3',lw=2,color='k'))
ax[axind].text(Tdec-0.5,para['T']+1,r'$1$')
ax[axind].text(para['T']-0.5,para['T']+1,r'$0$')
ax[axind].text(arrow_end[0],para['T']+1,r'$\alpha$')
trial_end=Tdec+(1-alp)*(para['T']-Tdec)
ax[axind].plot([0,trial_end],[0]*2,'k:')
ax[axind].fill_between([trial_end,trial_end+para['T_ITI']],[-4]*2,[4.5]*2,color=[0.5]*3)
ax[axind].text(Tdec+(1-alp)*(para['T']-Tdec)+1,-2,"inter-trial \n interval")
for t in np.arange(Tdec,para['T']):
    ax[axind].plot(Tdec+(1-alp)*(t-Tdec+1)*np.ones(t+2),np.arange(-(t+1),t+2,2),'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
ax[axind].plot(Nt_traj[:Tdec+1],'k.-',ms=1.5,lw=1)
ax[axind].plot(Tdec+(1-alp)*np.arange(para['T']-Tdec+1),Nt_traj[Tdec:],'k.-',ms=1.5,lw=1)

#formatting
ax[axind].set_ylim(-para['T']+1,para['T']+1)
ax[axind].set_xticks([0,Tdec,trial_end+para['T_ITI']])
ax[axind].set_yticks([])
ax[axind].set_xticklabels([r'$0$',r'$t_d$',r'$T$'])
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].spines['left'].set_visible(False)
ax[axind].set_xlabel('trial time')
ax[axind].set_ylabel(r'trial state, $s$')

#experiment response
axind=(1,1)

#response times over block
tmp=df_tmp.iloc[block_range[0]:block_range[1]]
ax[axind].plot(tmp.tDecision,'-',color=[0.7]*3)
ax[axind].plot(tmp[tmp.nPostInterval==150].tDecision,'.',color=colors[0])
ax[axind].plot(tmp[tmp.nPostInterval==50].tDecision,'.',color=colors[1])
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].spines['left'].set_visible(False)
ax[axind].plot([traj_it],[tmp.loc[traj_it].tDecision],'o',mec='k',mfc=colors[1])
ax[axind].set_yticks(range(2,12,2))
ax[axind].set_ylim(tlim)
ax[axind].set_xlabel('trial index')
ax[axind].set_ylabel('decision time')

axind=(2,1)
inds=df_tmp[cond].index.values
tStore=np.zeros(cond.sum())
fast_ind=np.zeros(cond.sum(),dtype=bool)
for it, ind in enumerate(inds[:-1]):
    late_Start=0
    if inds[it+1]-ind>late_Start:
        tStore[it]=df_tmp.iloc[ind+late_Start:inds[it+1]].tDecision.mean()
    fast_ind[it]=(df_tmp.iloc[ind].nPostInterval==50)
ax[axind].plot(tStore,'-',color=[0.7]*3)
ax[axind].plot(np.array(np.where(fast_ind))[0],tStore[fast_ind],'.',color=colors[1])
ax[axind].plot(np.array(np.where(~fast_ind))[0],tStore[~fast_ind],'.',color=colors[0])
ax[axind].plot([block_choice],[tStore[block_choice]],'o',mfc=colors[1],mec='k')
ax[axind].set_ylim(ax[1,1].get_ylim())
ax[axind].set_ylabel('block averaged decision time')
ax[axind].set_yticks(range(2,12,2))
ax[axind].set_ylabel('decision time')
ax[axind].set_ylim((4,14))
ax[axind].set_xlabel('block index')
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].spines['left'].set_visible(False)

axind=(0,1)
trial_time_vec=np.arange(len(df_mod.iloc[0].seq)+1)
sample_trial=df_mod.iloc[traj_it]
t_decision=sample_trial.tDecision
urgency=(sample_trial.rho_context-sample_trial.rho_long)*sample_trial.T_context+sample_trial.rho_long*trial_time_vec
for Nm in np.arange(para['T'],dtype=int):
    ax[axind].plot(np.arange(para['T']+1-Nm)+Nm,[get_pt_plus(t,Nt) for t,Nt in zip(np.arange(para['T']+1-Nm)+Nm,np.arange(para['T']+1-Nm)-Nm)],'-',ms=5,lw=0.5,color=[0.8]*3,mew=0.5,mfc='k',mec='k')
for Np in np.arange(para['T']):
    ax[axind].plot(np.arange(para['T']+1-Np)+Np,[1-get_pt_plus(t,Nt) for t,Nt in zip(np.arange(para['T']+1-Np)+Np,np.arange(para['T']+1-Np)-Np)],'-',ms=5,lw=0.5,color=[0.8]*3,mew=0.5,mfc='k',mec='k')
ax[axind].plot(trial_time_vec,urgency,'-',color=collong,lw=3,label='opportunity cost')
data=df_tmp.p_success.tolist()
mean_regret=1-np.mean(data,axis=0)
ax[axind].plot(np.arange(para['T']+1),mean_regret,'-',color=[0.8]*3,alpha=0.5,lw=3,label='avg. prediction regret')
regret=1-df_tmp.iloc[traj_it].p_success[:t_decision+1]
ax[axind].plot(range(t_decision+1),regret[:t_decision+1],'.-',color='k',lw=1,label='sample prediction regret',clip_on=False)
ax[axind].plot([0],[urgency[0]],'o',color=colcon,clip_on=False)

ax[axind].plot([t_decision],[regret[t_decision]],'k+',ms=10,clip_on=False)
ax[axind].set_xlabel('trial time')
ax[axind].set_ylim(0,0.5)
ax[axind].set_xticks((0,Tdec,para['T']))
ax[axind].set_xticklabels((r'$0$',r'$t_d$',r'$t_\textrm{max}$'))
ax[axind].set_xlim(0,para['T']+1)
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].grid('False')

axind=(0,2)
ax[axind].plot(df_mod.rho_context,color=colcon)
ax[axind].plot(df_mod.rho_long,color=collong)
ax[axind].set_xlabel('trial')
ax[axind].set_ylabel('filtered reward rate')
ax[axind].set_xlim(0,5000)#traj_it)
for condtmp in (df_tmp.index,df_tmp.nPostInterval==150,df_tmp.nPostInterval==50):
    ax[axind].plot(ax[axind].get_xlim(),[(df_tmp.loc[condtmp].nChoiceMade==df_tmp.loc[condtmp].nCorrectChoice).sum()/df_tmp.loc[condtmp].duration.sum()]*2,'k--')
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].set_ylim(0.035,0.055)

# left, bottom, width, height = [0.88, 0.92, 0.06, 0.06] #outset
# axin2b = fig.add_axes([left, bottom, width, height])
# left, bottom, width, height = [0.78, 0.92, 0.06, 0.06] #outset
# axin2a = fig.add_axes([left, bottom, width, height])
# cols=(colors[2],colors[4])
# for it,measure in enumerate(('rho_context','rho_long')):
#     timevec,data_store=get_transition_ensemble(df_tmp,measure)
#     for pit,data in enumerate(data_store):
#         me=np.mean(data,axis=0)
#         axin2b.plot(me[:-1],np.diff(me),color=cols[it])
#         axin2a.plot(timevec,me,color=cols[it])
# axin2b.set_yticks([])#yaxis.tick_right()
# axin2b.set_xticks([])#.xaxis.tick_top()
# axin2b.set_ylabel(r'$\dot{\rho}$')
# axin2b.set_xlabel(r'$\rho$')
# axin2b.spines['bottom'].set_position('zero')
# axin2b.spines['right'].set_visible(False)
# axin2b.spines['top'].set_visible(False)
# axin2b.spines['left'].set_visible(False)
# axin2a.set_yticks([])#yaxis.tick_right()
# axin2a.set_xticks([])#.xaxis.tick_top()
# axin2a.set_ylabel(r'$\rho$')
# axin2a.spines['bottom'].set_visible(False)
# axin2a.spines['right'].set_visible(False)
# axin2a.spines['top'].set_visible(False)
# axin2a.spines['left'].set_position('zero')


axind=(1,2)
coltmp=(colcon,collong)
bins=np.linspace(0.025,0.07,100)
counts,bins=np.histogram(df_tmp.trialRR.values,bins,weights=df_tmp.T_context.values,density=True)
ax[axind].plot(bins[:-1],counts,color=[0.8]*3,label=r'$\rho_\textrm{trial}$')
Tfast=df_tmp[df_tmp.nPostInterval==50].T_context.mean()
Tslow=df_tmp[df_tmp.nPostInterval==150].T_context.mean()
for it,condtmp in enumerate((df_tmp.nPostInterval==150,df_tmp.nPostInterval==50)):
    counts,bins=np.histogram(df_tmp.loc[condtmp].rho_context.values,bins,density=True)
    Ttmp=df_tmp.loc[condtmp].T_context.mean()
    ax[axind].plot(bins[:-1],Ttmp*counts/(Tfast+Tslow),'-',color=colors[it])
counts,bins=np.histogram(df_tmp.rho_context.values,bins,weights=df_tmp.T_context.values,density=True)
ax[axind].plot(bins[:-1],counts,color=colcon,label=r'$\rho_\textrm{context}$')
counts,bins=np.histogram(df_tmp.rho_long.values,bins,weights=df_tmp.T_context.values,density=True)
ax[axind].plot(bins[:-1],counts,color=collong,label=r'$\rho_\textrm{long}$')
ax[axind].set_yscale('log')
for condtmp in (df_tmp.index,df_tmp.nPostInterval==150,df_tmp.nPostInterval==50):
    ax[axind].plot([(df_tmp.loc[condtmp].nChoiceMade==df_tmp.loc[condtmp].nCorrectChoice).sum()/df_tmp.loc[condtmp].duration.sum()]*2,ax[axind].get_ylim(),'k--')
ax[axind].set_xlim(0.034,0.061)
ax[axind].legend(frameon=False,loc=1,prop={'size':8})
ax[axind].set_ylabel('counts')
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)


left, bottom, width, height = [0.55, 0.26, 0.07, 0.07] #outset
axin = fig.add_axes([left, bottom, width, height])

rho_context=[df_mod[df_mod.nPostInterval==n].rho_context.mean() for n in (150,50)]#[0.035,0.046]
T_context=[df_mod[df_mod.nPostInterval==n].duration.mean() for n in (150,50)]#[0.035,0.046][24,16]
T_avg=df_mod.duration.mean()#sum(T_context)/2
R_context=[rho*T for rho,T in zip(rho_context,T_context)]
R_avg=sum([R*T/sum(T_context) for R,T in zip(R_context,T_context)])

trial_time_vec=np.arange(para['T']+1+para['T_ITI'])
axin.set_ylim(0.70,0.83)
axin.set_xlim((para['T']+1+para['T_ITI']-9,para['T']+1+para['T_ITI']-2))
axin.set_ylabel(r'$\langle R_k\rangle$')
axin.set_xlabel(r'$\langle T_k\rangle$')
axin.plot(T_context,R_context,':',color='k')
axin.plot([0.9*T_avg,1.1*T_avg],[0.9*R_avg,1.1*R_avg],'-',color='k')
for it,lbl in enumerate(blockname):
    axin.plot([T_context[it]],[R_context[it]],'o',color='C'+str(it),ms=4,label=lbl,zorder=4)
    axin.plot([0.9*T_context[it],1.1*T_context[it]],[0.9*R_context[it],1.1*R_context[it]],'-',color='C'+str(it),ms=4,zorder=4)
axin.plot([T_avg],[R_avg],'o',color='k',ms=4,label='time-averaged')


axind=(2,2)
error_store=np.load(data_dir+'error_store.npy')
error_store2=np.load(data_dir+'errorstore2.npy')
times_store=np.load(data_dir+'times_store.npy')
mrkvec=['o','s','^']
for it in range(3):
    step=100
    running_avg_diff=error_store[it]
    ax[axind].plot(times_store[it],running_avg_diff,'k')
    ax[axind].plot([times_store[it][-1]],running_avg_diff[-1],mrkvec[it],color=colors[6+it])
    ax[axind].set_xscale('log')
    ax[axind].set_yscale('log')
ax[axind].set_xlabel(r'time')
ax[axind].set_ylabel(r'running error, $E_t$')
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)

left, bottom, width, height = [0.82, 0.16, 0.07, 0.07] #outset
axin = fig.add_axes([left, bottom, width, height])
tau_long_vec=[1e3,1e4,1e5]
block_size_vec=[100,200,400]
for tit,tau_long in enumerate(tau_long_vec):
    for bit,block_size in enumerate(block_size_vec):
        axin.plot([tau_long/block_size],[error_store2[tit,bit]],'k.')
for it in range(3):
    step=100
    running_avg_diff=error_store[it]
    axin.plot([tau_long_vec[it]/300],[running_avg_diff[-1]],mrkvec[it],ms=4,color=colors[6+it],zorder=4)        
xvar=np.logspace(0,3,100)
axin.plot(xvar,0.6*xvar**(-1),'k-')
axin.set_xlabel(r'$\tau_\textrm{long}/T_\textrm{block}$')
axin.set_xscale('log')
axin.set_yscale('log')
axin.set_ylim(1e-3,1e-1)
axin.spines['right'].set_visible(False)
axin.spines['top'].set_visible(False)
axin.patch.set_alpha(0.5)

fig.tight_layout()
axind=(0,1)
ax[axind].legend(frameon=False,prop={'size':9},loc='lower center',bbox_to_anchor=[0.6,-0.05])

fig.savefig(fig_dir+'multiple_timescale_test.pdf', transparent=True,bbox_inches='tight',dpi=300)
# -

#block
df_tmp=df_mod
fig,ax=pl.subplots(figsize=(3,3))
ax=[ax]
axind=0
ax[axind].text(0.1,0.9,'block',transform=ax[axind].transAxes)
ax[axind].plot(1-df_tmp.iloc[:5000].nPostInterval/200,'k--')
alpvec=1-df_tmp.iloc[:5000].nPostInterval/200
ax[axind].plot(alpvec[alpvec==1/4],'.',color=colors[0])
ax[axind].plot(alpvec[alpvec==3/4],'.',color=colors[1])
# ax[axind].plot([traj_it],[1-df_tmp.iloc[traj_it].nPostInterval/200],'o',mec='k',mfc='None')
ax[axind].set_ylim(0,1)
ax[axind].set_yticks([0,0.25,0.75,1])
ax[axind].set_yticklabels([r'$0$',r'$1/4$',r'$3/4$',r'$1$'])
ax[axind].set_ylabel(r'incentive strength, $\alpha$')
ax[axind].set_xlabel('trial index')
ax[axind].spines['right'].set_visible(False)
ax[axind].spines['top'].set_visible(False)
ax[axind].spines['left'].set_visible(False)
# ax[axind].set_xlim(block_range)
fig.savefig('multiple_timescale_v3_block.pdf', transparent=True,bbox_inches='tight',dpi=300)

# +
fig,ax=pl.subplots(4,1,figsize=(2.5,3))
# for it,rangevals in enumerate(([[30050,30250],[26000,34000],[20,int(1e5)]])):
rho_c=[]
for condtmp in (df_tmp.index,df_tmp.nPostInterval==150,df_tmp.nPostInterval==50):
    rho_c.append((df_tmp.loc[condtmp].nChoiceMade==df_tmp.loc[condtmp].nCorrectChoice).sum()/df_tmp.loc[condtmp].duration.sum())
df_mod['rho_c']=0
df_mod.loc[df_mod.nPostInterval==150,'rho_c']=rho_c[1]
df_mod.loc[df_mod.nPostInterval==50,'rho_c']=rho_c[2]
rangevals=[30000,31200]
trials=np.arange(rangevals[0],rangevals[1])
trial_noise=np.random.normal(0,0.004*(1-df_mod.iloc[rangevals[0]:rangevals[1]].nPostInterval/200))
ax[2].plot(trials,trial_noise)
# ax[2].plot(df_mod.iloc[rangevals[0]:rangevals[1]].trialRR-df_mod.iloc[rangevals[0]:rangevals[1]].rho_c)
ax[2].set_ylim(-0.01,0.01)
ax[2].spines['bottom'].set_position('zero')
ax[2].set_ylabel(r'$\rho_{\textrm{trial},k}-\rho_\alpha$')
ax[2].spines['bottom'].set_position('zero')
ax[2].set_yticks([-0.01,0.01])
ax[1].plot(df_mod.iloc[rangevals[0]:rangevals[1]].rho_c-rho_c[0])
ax[1].set_ylim(-0.01,0.01)
ax[1].set_ylabel(r'$\rho_\alpha-\rho$')
ax[1].spines['bottom'].set_position('zero')
ax[1].set_yticks([-0.01,0.01])


ax[0].plot(trials,rho_c[0]*np.ones(len(trials)))
ax[0].set_ylim(0.03,0.06)
ax[0].set_ylabel(r'$\rho$')
ax[0].spines['bottom'].set_visible(False)
ax[0].set_yticks([rho_c[0]-0.01,rho_c[0]+0.01])

ax[3].plot(trials,df_mod.iloc[rangevals[0]:rangevals[1]].rho_c+trial_noise)
ax[3].set_ylim(0.03,0.06)
ax[3].set_yticks([rho_c[0]-0.01,rho_c[0]+0.01])
ax[3].spines['bottom'].set_visible(False)

for axt in ax:
    axt.grid('off')
    axt.spines['right'].set_visible(False)
    axt.spines['top'].set_visible(False)
    axt.set_xticks([])
#     axt.set_yticks([0.041,0.045])
    axt.set_yticklabels('')
    
    
fig.subplots_adjust(hspace=0.1)
# fig.tight_layout()
fig.savefig(fig_dir+'timescale_stack.pdf', transparent=True,bbox_inches='tight',dpi=300)
# -

# ## Response analysis on Thura data

from lib.filter_plotting import get_transition_ensemble
from lib.lib import load_data

# Import data

# +
#local import

# import scipy.io as spio
# mat = spio.loadmat('../../exp_data/Thura_etal_2016/toktrials.mat', squeeze_me=True)
# col_names=mat['toktrials'].flatten().dtype.names
# df_data=pd.DataFrame(columns=col_names)
# for col_name in col_names:
#     df_data[col_name]=mat['toktrials'].flatten()[0][col_name]

# +
if not os.path.exists(os.getcwd()+'/../../exp_data/Thura_etal_2016/df_data_tdec_as_int.pkl'):
    df_data=load_data()
# df_data.to_pickle('../../exp_data/Thura_etal_2016/df_data.pkl')
    df_data.to_pickle('../../exp_data/Thura_etal_2016/df_data_tdec_as_int.pkl')

else:
#     df_data=pd.read_pickle('../../exp_data/Thura_etal_2016/df_data.pkl')
    df_data=pd.read_pickle('../../exp_data/Thura_etal_2016/df_data_tdec_as_int.pkl')
# subject=2
# df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)
# -

model_paras_S1_taus={'tau_context':407.047901,#/average_trial_duration,
            'sense_power':8.839935, 
            'tau_long':31620.805644,
            }
model_paras_S2_taus={'tau_context':1.40574372e+02, 
        'sense_power':-8.31123820e+00, 
        'tau_long':4.60248743e+04,
        }
model_paras_vec=[model_paras_S1_taus,model_paras_S2_taus]
sub_block_vec=[0,1]
for subject in [1,2]:
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)

    model_paras=model_paras_vec[subject-1]

    df_mod=get_model_output(df_act,model_paras,seed=10)
    
    plot_policies({'act':df_act,'mod':df_mod})#,file_name='subject_'+str(subject))

    fig,ax=pl.subplots(2,3,figsize=(8,5.5))
    title_str=['subject','fitted PGD agent']
    
    for tit, df_tmp in enumerate((df_act,df_mod)):

        df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff(periods=1) #dx_i=x_i-x_i-1
        df_tmp.loc[0,'nPostInterval_diff']=0 #replace nan ###remove initial block (an outlier in length)
        cond=df_tmp['nPostInterval_diff']!=0 #location of trial type switch,i.e. block start
        df_tmp['block_len']=0
        block_len_range=[-1,np.Inf]
        df_tmp.loc[cond,'block_len']=np.diff(np.insert(df_tmp[cond].index.values,cond.sum(),len(df_tmp))) #tag block starts with corresponding block length.np.diff uses period=-1 so add last block.
        nblocks=(df_tmp.block_len>0).sum()

        block_choice=int(nblocks/2)+sub_block_vec[subject-1]
        block_ind=df_tmp.loc[cond].index.values[block_choice]
        margin_size=20
        block_range=(block_ind-margin_size,block_ind+df_tmp.loc[block_ind].block_len+margin_size)
        traj_it=block_ind+int(df_tmp.loc[block_ind].block_len/2) #set as half way through block
        alp=1-df_tmp.loc[traj_it].nPostInterval/200
#         Nt_traj=df_tmp.loc[traj_it].Nt

        Tdec=df_tmp.loc[traj_it].tDecision
        
        if tit==0:
            axind=(1,0)
            ax[axind].plot(df_tmp[cond].block_len.values,'-',color=[0.7]*3)
            tmp=df_tmp[cond].reset_index(drop=True)
            for it in range(2):
                ax[axind].plot(tmp[tmp.nPostInterval==block_times[it]].block_len,'.',color=colors[it],label=blockname[it],ms=1.5)
            ax[axind].plot([block_choice],[df_tmp.loc[block_ind].block_len],'o',mec='k',mfc=colors[1])
            ax[axind].set_ylim(0,2*block_size)#2000)
            ax[axind].set_ylabel('block length')
            ax[axind].set_xlabel('block index')
            ax[axind].spines['right'].set_visible(False)
            ax[axind].spines['top'].set_visible(False)
            ax[axind].spines['left'].set_visible(False)

        axind=(0,tit+1)

        #response times over block
        tmp=df_tmp.iloc[block_range[0]:block_range[1]]
        ax[axind].plot(tmp.tDecision,'-',color=[0.7]*3)
        ax[axind].plot(tmp[tmp.nPostInterval==150].tDecision,'.',color=colors[0],ms=1.5)
        ax[axind].plot(tmp[tmp.nPostInterval==50].tDecision,'.',color=colors[1],ms=1.5)
        ax[axind].spines['right'].set_visible(False)
        ax[axind].spines['top'].set_visible(False)
        ax[axind].spines['left'].set_visible(False)
        ax[axind].plot([traj_it],[tmp.loc[traj_it].tDecision],'o',mec='k',mfc=colors[1])
        ax[axind].set_yticks(range(0,14,2))
        ax[axind].set_ylim(0,14)
        ax[axind].set_xlabel('trial index')
        ax[axind].set_ylabel('decision time')
        ax[axind].set_title(title_str[tit])
        
        axind=(1,tit+1)
        inds=df_tmp[cond].index.values
        tStore=np.zeros(cond.sum())
        fast_ind=np.zeros(cond.sum(),dtype=bool)
        for it, ind in enumerate(inds[:-1]):
            late_Start=0
            if inds[it+1]-ind>late_Start:
                tStore[it]=df_tmp.iloc[ind+late_Start:inds[it+1]].tDecision.mean()
            fast_ind[it]=(df_tmp.iloc[ind].nPostInterval==50)
        ax[axind].plot(tStore,'-',color=[0.7]*3)
        ax[axind].plot(np.array(np.where(fast_ind))[0],tStore[fast_ind],'.',color=colors[1],ms=1.5)
        ax[axind].plot(np.array(np.where(~fast_ind))[0],tStore[~fast_ind],'.',color=colors[0],ms=1.5)
        ax[axind].plot([block_choice],[tStore[block_choice]],'o',mfc=colors[1],mec='k')
        ax[axind].set_ylim(ax[1,1].get_ylim())
        ax[axind].set_ylabel('block-avg. decision time')
        ax[axind].set_yticks(range(0,14,2))
        ax[axind].set_ylim((0,14))
        ax[axind].set_xlabel('block index')
        ax[axind].spines['right'].set_visible(False)
        ax[axind].spines['top'].set_visible(False)
        ax[axind].spines['left'].set_visible(False)
    ax[0,0].axis('off')
    fig.tight_layout()
    fig.savefig('block_dynamics_subject_model_comp_'+str(subject)+'.pdf', transparent=True,bbox_inches='tight',dpi=300)

model_paras_S1_taus={'tau_context':338.226944,#/average_trial_duration,
            'sense_power':8.306030, 
            'tau_long':32623.003499,
            }
for subject in [1,2]:
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)
    model_paras=model_paras_S1_taus if subject==1 else model_paras_S2_taus
    df_mod=get_model_output(df_act,model_paras,seed=10)
    plot_transitions({'act':df_act,'mod':df_mod})

# Note that the block lengths differ for the two subjects and thus add to any inter subject varibability

# +
fig,ax=pl.subplots(2,2,figsize=(8,8))
figb,axb=pl.subplots(4,2,figsize=(8,12))
avg_blocksize=[]
import matplotlib.colors as mcolors
for sit, subject in enumerate(range(1,3)):
    df_tmp=df_data[df_data.idSubject==subject].reset_index(drop=True)
    df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff()
    cond=df_tmp['nPostInterval_diff']!=0   
    diffinds=np.diff(df_tmp[cond].index.values)
    h=ax[0,sit].hist2d(diffinds[1:], diffinds[:-1], bins=range(0,300,2),norm=mcolors.PowerNorm(0.5));
    ax[0,sit].set_title('subject '+str(subject))
    fig.colorbar(h[3],ax=ax[0,sit])
    ax[0,sit].set_xlabel('run length of block $i$')
    ax[0,sit].set_ylabel('run length of block $i+1$')
    axb[0,sit].set_ylim(0,500)
    axb[1,sit].set_xlabel('trial block index')
    axb[0,0].set_ylabel('block length')
   
    avg_blocksize.append(np.mean(diffinds))
#     ax2.plot(np.arange(len(diffinds))/len(diffinds),np.sort(diffinds),'.')
    
    df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff(periods=1) #dx_i=x_i-x_i-1
    df_tmp.loc[0,'nPostInterval_diff']=0 #replace nan ###remove initial block (an outlier in length)
    cond=df_tmp['nPostInterval_diff']!=0 #location of trial type switch,i.e. block start
    df_tmp['block_len']=0
    block_len_range=[-1,np.Inf]
    df_tmp.loc[cond,'block_len']=np.diff(np.insert(df_tmp[cond].index.values,cond.sum(),len(df_tmp))) #tag block starts with corresponding block length.np.diff uses period=-1 so add last block.

    inds=df_tmp[df_tmp.block_len!=0].index.values
    tStore=np.zeros(cond.sum())
    fast_ind=np.zeros(cond.sum(),dtype=bool)
    for it, ind in enumerate(inds[:-1]):
        late_Start=0
        if inds[it+1]-ind>late_Start:
            tStore[it]=df_tmp.iloc[ind+late_Start:inds[it+1]].tDecision.mean()
        fast_ind[it]=(df_tmp.iloc[ind].nPostInterval==50)
#     for it in range(0,len(inds[:-1]),10):
#         tStore[it:it+20]-=np.mean(tStore[it:it+20][~fast_ind[it:it+20]])-8
    axb[1,sit].plot(tStore,'-',color=[0.7]*3)
    axb[1,sit].plot(np.diff(tStore)/3,'-',color=[0.7]*3)
    axb[1,sit].plot(np.array(np.where(fast_ind))[0],tStore[fast_ind],'C1.')
    axb[1,sit].plot(np.array(np.where(~fast_ind))[0],tStore[~fast_ind],'C0.')
    axb[1,sit].set_ylim(-2.5,10)
    axb[1,0].set_ylabel('block averaged decision time')
    axb[1,1].plot([250,360],[8]*2,'k-',lw=5)

    axcum_ind=3
    tmp=df_tmp[cond & (df_tmp['block_len']>=block_len_range[0]) & (df_tmp['block_len']<=block_len_range[1])].block_len.values
    p=axb[axcum_ind,0].plot(np.arange(len(tmp))/len(tmp),np.sort(tmp),label='subject '+str(subject))#lower bound on block size
    axb[axcum_ind,0].axhline(np.mean(tmp),ls=':',color=p[-1].get_color())
    
    tmp=df_tmp[cond & (df_tmp['block_len']>=0)].reset_index(drop=True)
    axb[0,sit].plot(tmp.block_len.values,'-',color=[0.7]*3)
    axb[0,sit].plot(tmp[(tmp.nPostInterval==50)].block_len,'C1.',label='fast block')
    axb[0,sit].plot(tmp[(tmp.nPostInterval==150)].block_len,'C0.',label='slow block')
    
    axhist_ind=2
    counts,bins=np.histogram(tmp[(tmp.nPostInterval==150)].block_len,bins=np.linspace(0,1000,200))
    axb[axhist_ind,sit].plot(bins[:-1],counts,'C0-')
    counts,bins=np.histogram(tmp[(tmp.nPostInterval==50)].block_len,bins=np.linspace(0,1000,200))
    axb[axhist_ind,sit].plot(bins[:-1],counts,'C1-')
    axb[axhist_ind,sit].set_ylim(0,30)
    axb[axhist_ind,sit].set_xlabel('block length')
    axb[axhist_ind,sit].set_ylabel('count')
    
    tmp=df_tmp[(df_tmp.nPostInterval==50) & cond & (df_tmp['block_len']>=block_len_range[0]) & (df_tmp['block_len']<=block_len_range[1])].block_len.values
    axb[axcum_ind,1].plot(np.arange(len(tmp))/len(tmp),np.sort(tmp),'C'+str(sit)+':',label='fast blocks')#lower bound on block size
    
    tmp=df_tmp[(df_tmp.nPostInterval==150) & cond & (df_tmp['block_len']>=block_len_range[0]) & (df_tmp['block_len']<=block_len_range[1])].block_len.values
    axb[axcum_ind,1].plot(np.arange(len(tmp))/len(tmp),np.sort(tmp),'C'+str(sit)+'--',label='slow blocks')#lower bound on block size
    
    if sit==0:
        axb[axcum_ind,1].legend(frameon=False)
#     ax2.set_ylim(50,500)
    axb[axcum_ind,1].set_yscale('log')
    axb[axcum_ind,0].set_yscale('log')
axb[0,0].legend(frameon=False)
axb[0,0].set_title('subject 1')
axb[0,1].set_title('subject 2')
axb[0,1].legend(frameon=False)
axb[axcum_ind,0].legend(frameon=False)
axb[axcum_ind,0].set_ylabel('block length')
axb[axcum_ind,0].set_ylabel('block length')
axb[axcum_ind,0].set_xlabel('block rank (by length)')
axb[axcum_ind,1].set_xlabel('block rank (by length)')
fig.tight_layout()

#now do durations
for sit, subject in enumerate(range(1,3)):
    df_tmp=df_data[df_data.idSubject==subject].reset_index(drop=True)
    df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff()
    cond=df_tmp['nPostInterval_diff']!=0
    #sum durations between 
    inds=df_tmp[cond].index.values
    diffinds=[]
    for it,ind in enumerate(inds[:-1]):
        diffinds.append(df_tmp.iloc[ind:inds[it+1]].duration.sum())
#     diffinds=np.diff(df_tmp[cond].index.values)
    diffinds=np.array(diffinds)
    h=ax[1,sit].plot(diffinds[1:], diffinds[:-1],'.',ms=3.0,alpha=0.5);
#     ax[1,sit].set_title('subject '+str(subject))
#     fig.colorbar(h[3],ax=ax[0,sit])
    ax[1,sit].set_xlabel('duration of block $i$')
    ax[1,sit].set_ylabel('duration of block $i+1$')
    ax[1,sit].set_xlim(0,10000)
    ax[1,sit].set_ylim(0,10000)
    avg_blocksize.append(np.mean(diffinds))
#     ax2.plot(np.arange(len(diffinds))/len(diffinds),np.sort(diffinds),'.')
    
#     df_tmp['nPostInterval_diff']=df_tmp.nPostInterval.diff(periods=1) #dx_i=x_i-x_i-1
#     df_tmp.loc[0,'nPostInterval_diff']=0 #replace nan ###remove initial block (an outlier in length)
#     cond=df_tmp['nPostInterval_diff']!=0 #location of trial type switch,i.e. block start
#     df_tmp['block_len']=0
#     block_len_range=[-1,np.Inf]
#     df_tmp.loc[cond,'block_len']=np.diff(np.insert(df_tmp[cond].index.values,cond.sum(),len(df_tmp))) #tag block starts with corresponding block length.np.diff uses period=-1 so add last block.
    
#     tmp=df_tmp[cond & (df_tmp['block_len']>=block_len_range[0]) & (df_tmp['block_len']<=block_len_range[1])].block_len.values
#     p=ax[1,0].plot(np.arange(len(tmp))/len(tmp),np.sort(tmp),label='subject '+str(subject))#lower bound on block size
#     ax[1,0].axhline(np.mean(tmp),ls=':',color=p[-1].get_color())
#     tmp=df_tmp[(df_tmp.nPostInterval==50) & cond & (df_tmp['block_len']>=block_len_range[0]) & (df_tmp['block_len']<=block_len_range[1])].block_len.values
#     ax[1,1].plot(np.arange(len(tmp))/len(tmp),np.sort(tmp),'C'+str(sit)+':',label='fast blocks')#lower bound on block size
    
#     tmp=df_tmp[(df_tmp.nPostInterval==150) & cond & (df_tmp['block_len']>=block_len_range[0]) & (df_tmp['block_len']<=block_len_range[1])].block_len.values
#     ax[1,1].plot(np.arange(len(tmp))/len(tmp),np.sort(tmp),'C'+str(sit)+'--',label='slow blocks')#lower bound on block size
#     if sit==0:
#         ax[1,1].legend(frameon=False)
# #     ax2.set_ylim(50,500)
#     ax[1,1].set_yscale('log')
#     ax[1,0].set_yscale('log')
ax[0,0].legend(frameon=False)
fig.tight_layout()
figb.tight_layout()
# figb.savefig('block_stats.png', transparent=True,bbox_inches='tight',dpi=300)
# fig.savefig('block_return_stats.png', transparent=True,bbox_inches='tight',dpi=300)


# + {"code_folding": [0]}
def move_axes(ax, fig, subplot_spec=111):
      """Move an Axes object from a figure to a new pyplot managed Figure in
      the specified subplot."""
      # get a reference to the old figure context so we can release it
      old_fig = ax.figure
      # remove the Axes from it's original Figure context
      ax.remove()

      # set the pointer from the Axes to the new figure
      ax.figure = fig

      # add the Axes to the registry of axes for the figure
      fig.axes.append(ax)
      # twice, I don't know why...
      fig.add_axes(ax)

      # then to actually show the Axes in the new figure we have to make
      # a subplot with the positions etc for the Axes to go, so make a
      # subplot which will have a dummy Axes
      dummy_ax = fig.add_subplot(subplot_spec)

      # then copy the relevant data from the dummy to the ax
      ax.set_position(dummy_ax.get_position())

      # then remove the dummy
      dummy_ax.remove()

      # close the figure the original axis was bound to
      pl.close(old_fig)


# -

# Sense model (supp. fig.4)

# +

fig,axb=pl.subplots(1,2,figsize=(6,3))
# plot_transitions({'act':df_act,'mod':df_mod},ax=axb[2])
timevec=np.linspace(0,60)
axb[0].plot(timevec,1/timevec)
axb[0].set_xlim(0,60)
axb[0].set_ylim(0,0.8)
axb[0].set_ylabel('attentional cost rate, $q$')
axb[0].set_xlabel(r'tracking timescale, $T_\textrm{track}$')
axb[0].spines['right'].set_visible(False)
axb[0].spines['top'].set_visible(False)
for nu in [2,4,8]:
    axb[1].plot(timevec/30,1/(1+np.power(timevec/30,nu)),clip_on=False,label=r'$\nu='+str(nu)+r'$')
axb[1].set_xlabel(r'tracking cost, $Q=T_\textrm{sys}/T_\textrm{track}$')
axb[1].set_ylabel(r'$\tau_\textrm{context}$')
axb[1].set_yticks([0,1])
axb[1].set_yticklabels([r'$0$',r'$\tau_0$'])
axb[1].set_xticks([0,1])
axb[1].set_xticklabels([r'$0$',r'$1$'])
axb[1].set_xlim(0,2)
axb[1].set_ylim(0,1)
axb[1].spines['right'].set_visible(False)
axb[1].spines['top'].set_visible(False)
axb[1].legend(frameon=False)
# axb[0].axis('off')
# fig2=pl.figure()
# fig2.axes.append(axb[0,:])
# 2,axb2=pl.subplots(1,3,figsize=(8,3))
# axb2=
# move_axes(axb[0,:],fig2)
fig.tight_layout()
fig.savefig(fig_dir+'sense_model.pdf', transparent=True,bbox_inches='tight',dpi=300)
# -

# ## run PGD model

root_filename='v12uniconv_opt_neldermead_both_subject1'
subject_id=1
run_evaluation('opt',subject_id,df_data,data_dir,root_filename,model_type='default')

root_filename='v12uniconv_opt_neldermead_both_subject2'
subject_id=2
run_evaluation('opt',subject_id,df_data,data_dir,root_filename,model_type='default')

from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import inset_locator

data_dict={}
for subject in [1,2]:
    data_dict[subject]={}
    df_act=df_data[df_data.idSubject==subject].tail(int(1e5)).reset_index(drop=True)
    df_act['reward']=(df_act.nChoiceMade==df_act.nCorrectChoice)
    for npost in [150,50]:
#         print(df_act[df_act.nPostInterval==npost].reward.mean())
#         print(df_act[df_act.nPostInterval==npost].duration.mean())
        data_dict[subject][npost]=df_act[df_act.nPostInterval==npost].reward.mean()/df_act[df_act.nPostInterval==npost].duration.mean()
        print(data_dict[subject][npost])
np.save(data_dir+'PGD_reward_rates.npy',data_dict)

# +
fig,ax=pl.subplots(2,3,figsize=(8,6))
average_trial_duration=22
avg_blocksize=100

model_paras_S1_taus={'tau_context':646,
            'sense_power':11.703220, 
            'tau_long':45248,
                     'unitconv':1.0386
            }
model_paras_S2_taus={'tau_context':188,
            'sense_power':2.308, 
            'tau_long':56605,
                     'unitconv':1.15
            }


for sit,model_paras in enumerate([model_paras_S1_taus,model_paras_S2_taus]):
    df_act=df_data[df_data.idSubject==sit+1].reset_index(drop=True)    
    df_mod=get_model_output(df_act,model_paras)
    df_mod
    axtmp=plot_transitions({'act':df_act,'mod':df_mod},ax=ax[0,sit],col='C'+str(sit))
    axtmp.legend(frameon=False,loc=3 if sit==0 else 5)
    ax[0,sit].set_ylim(4,8.5)
#     axp=plot_transitions({'act':df_act,'mod':df_mod},ax=axs[sit])
#     axp.set_ylabel('average decision time')
# axp.legend(frameon=False)
    ax[0,sit].set_title('subject '+str(1+sit))
    
paranames=['tau_long','unitconv','sense_power']
for mode,para_name in enumerate([r'$\hat{\tau}_\textrm{long}$',r'reward bias, $\hat{K}^*$',r'tracking sensitivity, $\hat{\nu}^*$']):
    
    for sit,model_paras in enumerate([model_paras_S1_taus,model_paras_S2_taus]):
        error_grid=np.load('../../output/data/error_grid_v14uniconv_opt_neldermead_both_subject'+str(sit+1)+'_p'+str(mode)+'.npy')
        error_grid[error_grid==0]=np.nan
        paras=np.load('../../output/data/error_grid_v14uniconv_opt_neldermead_both_subject'+str(sit+1)+'_p'+str(mode)+'_pvecs.npy')
        tau_shortvec=paras[0]
        paravec2=paras[1]
        if mode==0:
            paravec2=np.log10(paravec2)
        optinds=np.unravel_index(np.nanargmin(error_grid),(len(tau_shortvec),len(paravec2)))

        x,y=np.meshgrid(np.log10(tau_shortvec),paravec2)
        min_err=np.nanmin(error_grid)
        max_err=np.nanmax(error_grid)
        ax[1,mode].contour(x,y,error_grid.T,levels=np.linspace(min_err,min_err+0.15,10),linewidths=1,colors='C'+str(sit),linestyles='-',)
#         ax[1,mode].contour(x,y,error_grid.T,levels=np.linspace(min_err,min_err+0.05*(max_err-min_err),10),linewidths=1,colors='C'+str(sit),linestyles='-',)

#         ax[1,mode].contour(x,y,error_grid.T,levels=40,linewidths=1,colors='C'+str(sit),linestyles='-')
#         ax[1,mode].plot([np.log10(tau_shortvec[optinds[0]])],[paravec2[optinds[1]]],'o')
        ax[1,mode].set_aspect('auto')

        ax[1,mode].set_xticks(np.arange(1,5))
        ax[1,mode].set_xticklabels([r'$10^'+str(n)+'$' for n in np.arange(1,5)])
        ax[1,mode].set_xlabel(r'$\hat{\tau}_\textrm{context}$')
        ax[1,mode].set_ylabel(para_name)
        ax[1,mode].set_xlim(1,4)
        if mode==0:
            ax[1,mode].plot([np.log10(model_paras['tau_context'])],[np.log10(model_paras[paranames[mode]])],'o',color='C'+str(sit))
        else:
            ax[1,mode].plot([np.log10(model_paras['tau_context'])],[model_paras[paranames[mode]]],'o',color='C'+str(sit))

ax[1,0].set_yticks(np.arange(2,6))
ax[1,0].set_yticklabels([r'$10^'+str(n)+'$' for n in np.arange(2,6)])
ax[1,0].set_ylim(3,5.5)
ax[1,0].fill_between(ax[1,0].get_xlim(),[5+np.log10(2)]*2,y2=[ax[1,0].get_ylim()[1]]*2,color=[0.8]*3,zorder=0)

# ax[1,1].set_xlim(0,800)
# ax[1,1].set_xticks([0,200,400,600,800])
ax[1,1].set_ylim(0.9,1.3)
# ax[1,2].set_xticks([0,200,400,600,800])
# ax[1,2].set_xlim(0,800)
ax[1,2].set_ylim(-5,20)

ax[0,2].axis('off')
# pl.subplots_adjust(wspace=0.5,hspace=0.5)

fig.tight_layout()
fig.savefig(fig_dir+'model_fitting.pdf', transparent=True,bbox_inches="tight",dpi=300)
# -


model_paras_S1_taus={'tau_context':646,
            'sense_power':11.703220, 
            'tau_long':45248,
                     'unitconv':1.0386
            }
model_paras_S2_taus={'tau_context':188,
            'sense_power':2.308, 
            'tau_long':56605,
                     'unitconv':1.15
            }

model_paras_vec=[model_paras_S1_taus,model_paras_S2_taus]
sub_block_vec=[0,1]
for subject in [1,2]:
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)

    model_paras=model_paras_vec[subject-1]

    df_mod=get_model_output(df_act,model_paras)
    
    plot_policies({'act':df_act,'mod':df_mod},file_name='subject_'+str(subject))

model_paras_vec=[model_paras_S1_taus,model_paras_S2_taus]
sub_block_vec=[0,1]
for subject in [1,2]:
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)

    model_paras=model_paras_vec[subject-1]

    df_mod=get_model_output(df_act,model_paras)
    
    plot_policies({'act':df_act,'mod':df_mod},file_name='subject_'+str(subject))

# ## neural urgency comparison

import scipy.io as spio


store_means = spio.loadmat('../../exp_data/Thura_etal_2016/store_mean.mat', squeeze_me=True)['store_mean']
store_ci = spio.loadmat('../../exp_data/Thura_etal_2016/store_ci_both_subjects.mat', squeeze_me=True)['store_ci']
store_times = spio.loadmat('../../exp_data/Thura_etal_2016/store_time.mat', squeeze_me=True)['store_time']
# col_names=mat['toktrials'].flatten().dtype.names

# +
# opt_paras=np.load('../../output/data/opt_paras_v9_opt_neldermead_finejusttaus_1_1000.npy')
# model_paras_S1_taus={'tau_context':opt_paras[0],  
#                 'tau_context_plus':opt_paras[0], 
#                 'tau_long':opt_paras[1],
#                 'Tcontext_bias':0,
#                 'shared_noise_variance_factor':0,
#                 'tau_shared_noise':1,
#                 'context_noise_bias':0,
#                 'context_noise_variance_factor':0
#                 }
# model_paras_S1_taus={'tau_context':407.047901,#/average_trial_duration,
#                 'sense_power':8.839935, 
#                 'tau_long':31620.805644,
#                 }
# model_paras_S1_taus={'tau_context':679.00308052 ,
#                 'tau_long':11741.3107371
#                 }
model_paras=[model_paras_S1_taus,model_paras_S2_taus]
trial_time_vec=np.arange(para['T']+1)
cmapvec=['Blues','YlOrRd']
fig,ax=pl.subplots(1,1,figsize=(3,3))
strvec=['slow block','fast block']
data_store=(slow_dat,fast_dat)
ax2=ax.twinx()
tmax=12
# scale_fac=[1,0.4]
# offset=[0,-0.2]
scale_fac=[1,1]
offset=[0,0]
for sit,model_para in enumerate(model_paras):
    df_act=df_data[df_data.idSubject==sit+1].reset_index(drop=True)
    df_mod=get_model_output(df_act,model_para)
    for bit,blocktime in enumerate(block_times):
        tmp_df=df_mod[df_mod.nPostInterval==blocktime]
        urgency_arr=tmp_df.rho_long.values[:,np.newaxis]*trial_time_vec[np.newaxis,:]+((tmp_df.rho_context.values-tmp_df.rho_long.values)*tmp_df.T_context.values)[:,np.newaxis]
        lowb=-0.55
        highb=0.85
        ax.set_ylim(lowb,highb)
    #     binsvec=np.linspace(lowb,highb,100)
    #     hist_data=np.zeros((len(binsvec)-1,len(trial_time_vec)))
    #     for tit in trial_time_vec:
    #         counts,bins=np.histogram(urgency_arr[:,tit],bins=binsvec)
    #         hist_data[:,tit]=counts
        mean_urg=np.mean(urgency_arr,axis=0)
        std_urg=np.std(urgency_arr,axis=0)
    #     ax.errorbar(x=np.arange(0,para['T']+1)*0.2+0.1,y=mean_urg,yerr=np.std(urgency_arr,axis=0),elinewidth=1.5,fmt='o',ms=0,ecolor='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])
        ax.fill_between(x=np.arange(0,para['T']+1),y1=scale_fac[sit]*(mean_urg-std_urg)+offset[sit],y2=scale_fac[sit]*(mean_urg+std_urg)+offset[sit],alpha=0.5,color='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])

        #     hist_data[hist_data<500]=np.nan
    #     ax.imshow(hist_data,origin='lower',extent=[0,(para['T']+1)*0.2,lowb, highb],cmap=cmapvec[bit],aspect=1.7)
        ax.legend(frameon=False,prop={'size': 12},loc=4)
        ax.set_ylabel('opportunity cost')
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
        ax.set_xlim(0,tmax)
        ax.set_xlabel(r'time, steps')
        max_rate=70
        error_bars=store_ci[sit,bit]-store_means[sit,bit][:,np.newaxis]
        error_bars[:,0]*=-1
        ax2.errorbar(x=store_times[sit,bit],y=store_means[sit,bit],yerr=error_bars.T,elinewidth=1.5,fmt='o',ms=2,ecolor='C'+str(bit),color='C'+str(bit),label=strvec[bit])#/np.sqrt(urgency_arr.shape[0])

#         ax2.plot(,store_means[sit,bit],'.:',mew=2,ms=2,mfc='None',color='C'+str(bit))
        ax2.set_ylim(0,40)
        ax2.set_ylabel('firing rate, Hz')
        ax.set_ylim([lowb,highb])
# ax2.legend(frameon=False,prop={'size': 10},loc=4)
ax.set_xticks((0,5,10))

fig.tight_layout()
fig.savefig('neuralurg_tokens.pdf', transparent=True,bbox_inches="tight",dpi=300)

# +
# opt_paras=np.load('../../output/data/opt_paras_v9_opt_neldermead_finejusttaus_1_1000.npy')
# model_paras_S1_taus={'tau_context':opt_paras[0],  
#                 'tau_context_plus':opt_paras[0], 
#                 'tau_long':opt_paras[1],
#                 'Tcontext_bias':0,
#                 'shared_noise_variance_factor':0,
#                 'tau_shared_noise':1,
#                 'context_noise_bias':0,
#                 'context_noise_variance_factor':0
#                 }
# model_paras_S1_taus={'tau_context':407.047901,#/average_trial_duration,
#                 'sense_power':8.839935, 
#                 'tau_long':31620.805644,
#                 }
# model_paras_S1_taus={'tau_context':679.00308052 ,
#                 'tau_long':11741.3107371
#                 }
model_paras=[model_paras_S1_taus,model_paras_S2_taus]
trial_time_vec=np.arange(para['T']+1)
cmapvec=['Blues','YlOrRd']
fig,ax=pl.subplots(1,2,figsize=(6,3))
strvec=['slow block','fast block']
data_store=(slow_dat,fast_dat)

tmax=12
# scale_fac=[1,0.4]
# offset=[0,-0.2]
scale_fac=[1,1]
offset=[0,0]
for sit,model_para in enumerate(model_paras):
    ax2=ax[sit].twinx()
    df_act=df_data[df_data.idSubject==sit+1].reset_index(drop=True)
    df_mod=get_model_output(df_act,model_para)
    for bit,blocktime in enumerate(block_times):
        tmp_df=df_mod[df_mod.nPostInterval==blocktime]
        urgency_arr=tmp_df.rho_long.values[:,np.newaxis]*trial_time_vec[np.newaxis,:]+((tmp_df.rho_context.values-tmp_df.rho_long.values)*tmp_df.T_context.values)[:,np.newaxis]
        if sit==0:
            lowb=-0.55
            highb=0.85
        else:
            lowb=-0.55
            highb=0.85
            
        ax[sit].set_ylim(lowb,highb)
    #     binsvec=np.linspace(lowb,highb,100)
    #     hist_data=np.zeros((len(binsvec)-1,len(trial_time_vec)))
    #     for tit in trial_time_vec:
    #         counts,bins=np.histogram(urgency_arr[:,tit],bins=binsvec)
    #         hist_data[:,tit]=counts
        mean_urg=np.mean(urgency_arr,axis=0)
        std_urg=np.std(urgency_arr,axis=0)
    #     ax.errorbar(x=np.arange(0,para['T']+1)*0.2+0.1,y=mean_urg,yerr=np.std(urgency_arr,axis=0),elinewidth=1.5,fmt='o',ms=0,ecolor='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])
        ax[sit].fill_between(x=np.arange(0,para['T']+1),y1=scale_fac[sit]*(mean_urg-std_urg)+offset[sit],y2=scale_fac[sit]*(mean_urg+std_urg)+offset[sit],alpha=0.5,color='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])

        #     hist_data[hist_data<500]=np.nan
    #     ax.imshow(hist_data,origin='lower',extent=[0,(para['T']+1)*0.2,lowb, highb],cmap=cmapvec[bit],aspect=1.7)
        ax[sit].legend(frameon=False,prop={'size': 12},loc=4)
        ax[sit].set_ylabel('opportunity cost')
        ax[sit].yaxis.grid(False)
        ax[sit].xaxis.grid(True)
        ax[sit].set_xlim(0,tmax)
        ax[sit].set_xlabel(r'time, steps')
        max_rate=70
        error_bars=store_ci[sit,bit]-store_means[sit,bit][:,np.newaxis]
        error_bars[:,0]*=-1
        ax2.errorbar(x=store_times[sit,bit],y=store_means[sit,bit],yerr=error_bars.T,elinewidth=1.5,fmt='o',ms=2,ecolor='C'+str(bit),color='C'+str(bit),label=strvec[bit])#/np.sqrt(urgency_arr.shape[0])

#         ax2.plot(,store_means[sit,bit],'.:',mew=2,ms=2,mfc='None',color='C'+str(bit))
        if sit==1:
            ax2.set_ylim(0,40)
        else:
            ax2.set_ylim(8,20)
            ax2.set_ylim(3,25)
        ax2.set_ylabel('firing rate, Hz')
        ax[sit].set_ylim([lowb,highb])
# ax2.legend(frameon=False,prop={'size': 10},loc=4)
        ax[sit].set_xticks((0,5,10))
    ax[sit].set_title('(n='+str(9 if sit==0 else 21)+')')

fig.tight_layout()
# fig.savefig('neuralurg_tokens.pdf', transparent=True,bbox_inches="tight",dpi=300)

# +
# opt_paras=np.load('../../output/data/opt_paras_v9_opt_neldermead_finejusttaus_1_1000.npy')
# model_paras_S1_taus={'tau_context':opt_paras[0],  
#                 'tau_context_plus':opt_paras[0], 
#                 'tau_long':opt_paras[1],
#                 'Tcontext_bias':0,
#                 'shared_noise_variance_factor':0,
#                 'tau_shared_noise':1,
#                 'context_noise_bias':0,
#                 'context_noise_variance_factor':0
#                 }
# model_paras_S1_taus={'tau_context':407.047901,#/average_trial_duration,
#                 'sense_power':8.839935, 
#                 'tau_long':31620.805644,
#                 }
# model_paras_S1_taus={'tau_context':679.00308052 ,
#                 'tau_long':11741.3107371
#                 }
model_paras=[model_paras_S1_taus,model_paras_S2_taus]
trial_time_vec=np.arange(para['T']+1)
cmapvec=['Blues','YlOrRd']
fig,ax=pl.subplots(1,2,figsize=(6,3))
strvec=['slow block','fast block']
data_store=(slow_dat,fast_dat)

tmax=12
# scale_fac=[1,0.4]
# offset=[0,-0.2]
scale_fac=[1,1]
offset=[0,0]
for sit,model_para in enumerate(model_paras):
    ax2=ax[sit].twinx()
    df_act=df_data[df_data.idSubject==sit+1].reset_index(drop=True)
    df_mod=get_model_output(df_act,model_para)
    for bit,blocktime in enumerate(block_times):
        tmp_df=df_mod[df_mod.nPostInterval==blocktime]
        urgency_arr=tmp_df.rho_long.values[:,np.newaxis]*trial_time_vec[np.newaxis,:]+((tmp_df.rho_context.values-tmp_df.rho_long.values)*tmp_df.T_context.values)[:,np.newaxis]
        if sit==0:
#             lowb=-0.55
#             highb=0.85
            lowb=-0.75
            highb=2
        else:
            lowb=-0.55
            highb=0.85
            
        ax[sit].set_ylim(lowb,highb)
    #     binsvec=np.linspace(lowb,highb,100)
    #     hist_data=np.zeros((len(binsvec)-1,len(trial_time_vec)))
    #     for tit in trial_time_vec:
    #         counts,bins=np.histogram(urgency_arr[:,tit],bins=binsvec)
    #         hist_data[:,tit]=counts
        mean_urg=np.mean(urgency_arr,axis=0)
        std_urg=np.std(urgency_arr,axis=0)
    #     ax.errorbar(x=np.arange(0,para['T']+1)*0.2+0.1,y=mean_urg,yerr=np.std(urgency_arr,axis=0),elinewidth=1.5,fmt='o',ms=0,ecolor='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])
        ax[sit].fill_between(x=np.arange(0,para['T']+1),y1=scale_fac[sit]*(mean_urg-std_urg)+offset[sit],y2=scale_fac[sit]*(mean_urg+std_urg)+offset[sit],alpha=0.5,color='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])

        #     hist_data[hist_data<500]=np.nan
    #     ax.imshow(hist_data,origin='lower',extent=[0,(para['T']+1)*0.2,lowb, highb],cmap=cmapvec[bit],aspect=1.7)
        ax[sit].legend(frameon=False,prop={'size': 12},loc=4)
        ax[sit].set_ylabel('opportunity cost')
        ax[sit].yaxis.grid(False)
        ax[sit].xaxis.grid(True)
        ax[sit].set_xlim(0,tmax)
        ax[sit].set_xlabel(r'time, steps')
        max_rate=70
        error_bars=store_ci[sit,bit]-store_means[sit,bit][:,np.newaxis]
        error_bars[:,0]*=-1
        ax2.errorbar(x=store_times[sit,bit],y=store_means[sit,bit],yerr=error_bars.T,elinewidth=1.5,fmt='o',ms=2,ecolor='C'+str(bit),color='C'+str(bit),label=strvec[bit])#/np.sqrt(urgency_arr.shape[0])

#         ax2.plot(,store_means[sit,bit],'.:',mew=2,ms=2,mfc='None',color='C'+str(bit))
        if sit==1:
            ax2.set_ylim(0,40)
        else:
            ax2.set_ylim(0,40)
#             ax2.set_ylim(8,20)
#             ax2.set_ylim(3,25)
        ax2.set_ylabel('firing rate, Hz')
        ax[sit].set_ylim([lowb,highb])
# ax2.legend(frameon=False,prop={'size': 10},loc=4)
        ax[sit].set_xticks((0,5,10))
    ax[sit].set_title('(n='+str(9 if sit==0 else 21)+')')

fig.tight_layout()
fig.savefig('neuralurg_tokens_separate.pdf', transparent=True,bbox_inches="tight",dpi=300)
# -

# ## FIg 6b

# +
model_paras=[model_paras_S1_taus,model_paras_S2_taus]
trial_time_vec=np.arange(para['T']+1)
cmapvec=['Blues','YlOrRd']
fig,ax=pl.subplots(1,1,figsize=(3,3))
strvec=['slow block','fast block']
data_store=(slow_dat,fast_dat)
ax2=ax.twinx()
tmax=12
for bit,blocktime in enumerate(block_times):
    store_rho_long=np.empty((0))
    store_rho_context=np.empty((0))
    store_T_context=np.empty((0))
    for sit,model_para in enumerate(model_paras):
        df_act=df_data[df_data.idSubject==sit+1].reset_index(drop=True)
        df_mod=get_model_output(df_act,model_para)
        tmp_df=df_mod[df_mod.nPostInterval==blocktime]
        store_rho_long=np.concatenate((store_rho_long,tmp_df.rho_long.values))
        store_rho_context=np.concatenate((store_rho_context,tmp_df.rho_context.values))
        store_T_context=np.concatenate((store_T_context,tmp_df.T_context.values))
        
    urgency_arr=store_rho_long[:,np.newaxis]*trial_time_vec[np.newaxis,:]+((store_rho_context-store_rho_long)*store_T_context)[:,np.newaxis]
#     urgency_arr=np.concatenate(urgency_arr)
    lowb=-0.6
    highb=0.97
    ax.set_ylim(lowb,highb)
#     binsvec=np.linspace(lowb,highb,100)
#     hist_data=np.zeros((len(binsvec)-1,len(trial_time_vec)))
#     for tit in trial_time_vec:
#         counts,bins=np.histogram(urgency_arr[:,tit],bins=binsvec)
#         hist_data[:,tit]=counts
    mean_urg=np.mean(urgency_arr,axis=0)
    std_urg=np.std(urgency_arr,axis=0)
#     ax.errorbar(x=np.arange(0,para['T']+1)*0.2+0.1,y=mean_urg,yerr=np.std(urgency_arr,axis=0),elinewidth=1.5,fmt='o',ms=0,ecolor='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])
    ax.fill_between(x=np.arange(0,para['T']+1),y1=mean_urg-std_urg,y2=mean_urg+std_urg,alpha=0.5,color='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])

    #     hist_data[hist_data<500]=np.nan
#     ax.imshow(hist_data,origin='lower',extent=[0,(para['T']+1)*0.2,lowb, highb],cmap=cmapvec[bit],aspect=1.7)
    ax.legend(frameon=False,prop={'size': 12},loc=4)
    ax.set_ylabel('opportunity cost')
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    ax.set_xlim(0,tmax)
    ax.set_xlabel(r'time, steps')
    max_rate=70
    if bit==1:
        ax2.plot(np.arange(1,11),data_store[bit][:-1,1],'.',mew=2,ms=5,color='C'+str(bit),label=strvec[bit])
        for pit,pair in enumerate(store_ci[bit]):
            ax2.plot([2*pit+2]*2,pair,'-',color='C'+str(bit),lw=1.5)
    else:
        ax2.plot(np.arange(1,11),data_store[bit][:-1,1],'.',mew=2,ms=5,color='C'+str(bit),label=strvec[bit])
        for pit,pair in enumerate(store_ci[bit]):
            ax2.plot([2*pit+2]*2,pair,'-',color='C'+str(bit),lw=1.)
    ax2.set_ylim(0,40)
    ax2.set_xlim(0,10.5)
    ax2.set_ylabel('firing rate, Hz')
    ax.set_ylim([lowb,highb])
ax2.legend(frameon=False,prop={'size': 10},loc=4)
ax.set_xticks(np.arange(0,11,2))

fig.tight_layout()
fig.savefig('neuralurg_tokens.pdf', transparent=True,bbox_inches="tight",dpi=300)
# -

#Digitized from Thura & Cisek J. Neuroscience fig.8b
fast_dat=np.asarray([[0.10000000000000009, 19.235021459227468],
[0.3186666666666669, 21.66266094420601],
[0.5106666666666668, 21.643090128755368],
[0.724, 22.010472103004297],
[0.9213333333333331, 22.506094420600864],
[1.124, 23.77442060085837],
[1.3159999999999998, 25.428669527897],
[1.5346666666666664, 25.04944206008584],
[1.7053333333333334, 28.376824034334767],
[1.94, 25.96377682403434],
[2.1159999999999997, 23.162575107296142]])
fast_dat_upper=np.asarray([[0.3179401993355482, 22.464436035882827],
[0.7272425249169436, 22.995365843288976],
[1.120598006644518, 25.11338803622295],
[1.5299003322259135, 27.35912588750478],
[1.9338870431893684, 28.91390672165299]])
fast_dat_lower=np.asarray([[0.3179401993355482, 20.85200459164151],
[0.7166112956810631, 20.973598061306916],
[1.120598006644518, 22.50278474554653],
[1.5299003322259135, 22.93133795331831],
[1.9338870431893684, 23.027252242676756]])
slow_dat=np.asarray([[0.10000000000000009, 13.930300429184552],
[0.27600000000000025, 15.558283261802579],
[0.5106666666666668, 16.98214592274678],
[0.6813333333333336, 18.455450643776828],
[0.9053333333333331, 19.080686695278974],
[1.092, 20.116738197424894],
[1.3053333333333335, 21.51416309012876],
[1.492, 22.781974248927042],
[1.7053333333333334, 23.0206008583691],
[1.8973333333333335, 24.67484978540773],
[2.1213333333333333, 29.085493562231765]])
slow_dat_lower=np.asarray([[0.2754152823920266, 15.170783555120956],
[0.6794019933554818, 17.928489434973002],
[1.0887043189368768, 19.45759108881425],
[1.4873754152823917, 21.882658050252964],
[1.891362126245847, 22.87436758641214]])
slow_dat_upper=np.asarray([[0.2807308970099669, 16.09208792143191],
[0.6847176079734221, 19.028952850644103],
[1.0833887043189367, 20.788571914459418],
[1.4873754152823917, 23.751030993580205],
[1.891362126245847, 26.585519323158028]])

# ## Statistics

# ## Supp FIg 6

# +
model_paras_S1_taus={'tau_context':646,
            'sense_power':11.703220, 
            'tau_long':45248,
                     'unitconv':1.0386
            }
model_paras_S2_taus={'tau_context':188,
            'sense_power':2.308, 
            'tau_long':56605,
                     'unitconv':1.15
            }


for sit,model_paras in enumerate([model_paras_S1_taus,model_paras_S2_taus]):
    df_act=df_data[df_data.idSubject==sit+1].reset_index(drop=True)    
    df_mod=get_model_output(df_act,model_paras)
    plot_stats({'act':df_act,'mod':df_mod},'stats_subj_'+str(sit+1))
# -

# ## Context unaware

root_filename='v13unaware_opt_neldermead_justfastslowmodel_subject1'
root_dir='../../output/data/'
run_evaluation('opt',1,df_data,root_dir,root_filename,model_type='slow_to_fast_only')

root_filename='v13unaware_opt_neldermead_justfastslowmodel_subject2'
root_dir='../../output/data/'
run_evaluation('opt',2,df_data,root_dir,root_filename,model_type='slow_to_fast_only')

# +
df_act=df_data[df_data.idSubject==1].reset_index(drop=True)

model_paras={'tau_long':340,'unitconv':1.03}

df_mod=get_model_output(df_act,model_paras)
plot_policies({'act':df_act,'mod':df_mod},file_name='subject_1')
# -

# ## Supp FIg. 7 C

# +
# opt_paras=np.load('../../output/data/opt_paras_v9_opt_neldermead_finejusttaus_1_1000.npy')
# model_paras_S1_taus={'tau_context':opt_paras[0],  
#                 'tau_context_plus':opt_paras[0], 
#                 'tau_long':opt_paras[1],
#                 'Tcontext_bias':0,
#                 'shared_noise_variance_factor':0,
#                 'tau_shared_noise':1,
#                 'context_noise_bias':0,
#                 'context_noise_variance_factor':0
#                 }

# model_paras=[model_paras_S1_taus,model_paras_S2_taus]
model_paras=[{'tau_long':78.653336,'unitconv':1.03},{'tau_long':45.4,'unitconv':1.45}]#_S1_taus,model_paras_S2_taus]

trial_time_vec=np.arange(para['T']+1)
cmapvec=['Blues','YlOrRd']
fig,ax=pl.subplots(1,1,figsize=(3,3))
strvec=['slow block','fast block']
data_store=(slow_dat,fast_dat)
ax2=ax.twinx()
tmax=12
for bit,blocktime in enumerate(block_times):
    store_rho_long=np.empty((0))
    store_rho_context=np.empty((0))
    store_T_context=np.empty((0))
    for sit,model_para in enumerate(model_paras):
#         if sit==0:
        df_act=df_data[df_data.idSubject==sit+1].reset_index(drop=True)
        df_mod=get_model_output(df_act,model_para)
        tmp_df=df_mod[df_mod.nPostInterval==blocktime]
        store_rho_long=np.concatenate((store_rho_long,tmp_df.rho_long.values))
        store_rho_context=np.concatenate((store_rho_context,tmp_df.rho_context.values))
        store_T_context=np.concatenate((store_T_context,tmp_df.T_context.values))
        
    urgency_arr=store_rho_long[:,np.newaxis]*trial_time_vec[np.newaxis,:]#+((store_rho_context-store_rho_long)*store_T_context)[:,np.newaxis]
#     urgency_arr=np.concatenate(urgency_arr)
    lowb=-0.45
    highb=0.97
    ax.set_ylim(lowb,highb)
#     binsvec=np.linspace(lowb,highb,100)
#     hist_data=np.zeros((len(binsvec)-1,len(trial_time_vec)))
#     for tit in trial_time_vec:
#         counts,bins=np.histogram(urgency_arr[:,tit],bins=binsvec)
#         hist_data[:,tit]=counts
    mean_urg=np.mean(urgency_arr,axis=0)
    std_urg=np.std(urgency_arr,axis=0)
#     ax.errorbar(x=np.arange(0,para['T']+1)*0.2+0.1,y=mean_urg,yerr=np.std(urgency_arr,axis=0),elinewidth=1.5,fmt='o',ms=0,ecolor='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])
    ax.fill_between(x=np.arange(0,para['T']+1),y1=mean_urg-std_urg,y2=mean_urg+std_urg,alpha=0.5,color='C'+str(bit))#/np.sqrt(urgency_arr.shape[0])

    #     hist_data[hist_data<500]=np.nan
#     ax.imshow(hist_data,origin='lower',extent=[0,(para['T']+1)*0.2,lowb, highb],cmap=cmapvec[bit],aspect=1.7)
    ax.legend(frameon=False,prop={'size': 12},loc=4)
    ax.set_ylabel('opportunity cost')
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    ax.set_xlim(0,tmax)
    ax.set_xlabel(r'time, steps')
    max_rate=70
    if bit==1:
        ax2.plot(np.arange(1,11),data_store[bit][:-1,1],'.',mew=2,ms=5,color='C'+str(bit),label=strvec[bit])
        for pit,pair in enumerate(store_ci[bit]):
            ax2.plot([2*pit+2]*2,pair,'-',color='C'+str(bit),lw=1.5)
    else:
        ax2.plot(np.arange(1,11),data_store[bit][:-1,1],'.',mew=2,ms=5,color='C'+str(bit),label=strvec[bit])
        for pit,pair in enumerate(store_ci[bit]):
            ax2.plot([2*pit+2]*2,pair,'-',color='C'+str(bit),lw=1.)
    ax2.set_ylim(0,40)
    ax2.set_xlim(0,10.5)
    ax2.set_ylabel('firing rate, Hz')
#     ax.set_ylim([lowb,highb])
ax2.legend(frameon=False,prop={'size': 10},loc=4)
ax.set_xticks(np.arange(0,11,2))

fig.tight_layout()
fig.savefig('neuralurg_tokens_trial_unaware.pdf', transparent=True,bbox_inches="tight",dpi=300)
