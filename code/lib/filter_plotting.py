#plotting
import matplotlib.pyplot as pl
import seaborn as sns
sns.set_style("ticks", {'axes.grid' : True})
pl.rc("figure", facecolor="white",figsize = (8,8))
pl.rc('text', usetex=True)
pl.rc('text.latex', preamble=[r'\usepackage{amsmath}'])
pl.rc('lines',markeredgewidth = 2)
pl.rc('font',size = 12)
from scipy.interpolate import griddata
import numpy as np
from scipy.stats import pearsonr
from lib.filter_lib import get_transition_ensemble
from mpl_toolkits.axes_grid1 import inset_locator
para=dict()
para['T']=15
para['T_ITI']=7.5 #para['T']/2=7.5 in primate experiments set to 8 so that all trial durations are integers
para['p']=1/2
para['tp']=0

def get_plots(df_dict,return_stats,return_policy,return_transitions):
    if return_transitions:
        plot_transitions(df_dict)
    if return_stats:
        plot_stats(df_dict)
    if return_policy:
        plot_policies(df_dict)
    return 0

def plot_stats(df_dict,save_name):
#     figt,axt=pl.subplots(1,1)
    fig,ax=pl.subplots(3,2)
    titlestrvec=('subject','model')
    data_type=('act','mod')
#     acorr=[]
    for j,dtype in enumerate(data_type):
        df_tmp=df_dict[dtype]
        win_size=1000
        feature='trialRR'
        me=df_tmp[feature].rolling(window=win_size).mean()
        std=df_tmp[feature].rolling(window=win_size).std()
        maxval=df_tmp[feature].rolling(window=win_size).max()
        minval=df_tmp[feature].rolling(window=win_size).min()

        ax[0,j].fill_between(range(len(me)), me - std, me + std,color='gray', alpha=0.2)
        df_tmp[feature].rolling(window=win_size).mean().plot.line(ax=ax[0,j],color='C'+str(j),label=titlestrvec[j])
        #ax[0,j].set_ylim(10,25)
        ax[0,j].set_xlim(0,80000)
        ax[0,0].set_ylabel(feature)
        ax[0,j].set_xlabel('trial')
        ax[0,j].legend(frameon=False)
        ax[0,j].fill_between(range(len(me)), me - std, me + std,color='b', alpha=0.2)

        df_tmp.duration.hist(ax=ax[1,1],alpha=0.5,bins=np.arange(25))#para['T']+2))
        ax[1,1].set_xlabel('decision time')
        ax[1,1].axvline(x=df_tmp.duration.mean(),ls='--',color='C'+str(j))
        df_tmp.trialRR.hist(ax=ax[1,0],alpha=0.5,bins=np.linspace(0.02,0.07,25))
        ax[1,0].set_xlabel('trial reward rate')
        ax[1,0].axvline(x=df_tmp.trialRR.mean(),ls='--',color='C'+str(j))
        ax[1,j].ticklabel_format(axis='y',scilimits=(0,0))
        auto_corr=np.correlate(df_tmp.duration.values, df_tmp.duration.values, mode='same')#/df_tmp.duration.var()
        ax[2,0].plot(np.arange(-len(auto_corr)/2,len(auto_corr)/2),auto_corr,label=titlestrvec[j])#/auto_corr[int(len(auto_corr)/2)])
    
    ax[2,0].set_xlim(1,len(auto_corr)/2)
    ax[2,0].set_xscale('log')
#         ax[2,j].set_yscale('log')
#         ax[2,j].set_ylim(0.95,1)
    ax[2,0].set_ylim(3.5e7,3.8e7)
    ax[2,0].set_xlabel('time lag')
    ax[2,0].set_ylabel('correlation')
    ax[2,0].legend(frameon=False)
#         acorr.append(auto_corr/auto_corr[int(len(auto_corr)/2)])
    
    #fig,ax=pl.subplots(figsize=(4,4))
    h=ax[2,1].hist2d(df_dict['act'].tDecision.values,df_dict['mod'].tDecision.values,bins=(np.arange(para['T']+1),np.arange(para['T'])),cmap='gray_r')
    cbar_ax=inset_locator.inset_axes(ax[2,1],width='50%',height='3%',loc=4)
    cb=pl.colorbar(h[3],cax=cbar_ax,orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    ax[2,1].text(1,12,r'Pearson $r='+str(round(np.corrcoef(df_dict['act'].tDecision.values,df_dict['mod'].tDecision.values)[0,1]*1000)/1000)+'$')
    ax[2,1].plot(ax[2,1].get_xlim(),ax[2,1].get_xlim(),'w--')
    ax[2,1].set_xlabel('model decision time')
    ax[2,1].set_ylabel('data decision time')
    ax[2,1].set_xticks([0,5,10,15])
    ax[2,1].set_yticks([0,5,10,15])
    ax[2,1].set_xlim(0,15)
    ax[2,1].set_ylim(0,15)
        #difference correlations
#         for b in block_times:
#             difft=df_tmp.tDecision[df_tmp.nPostInterval==b].diff()
#             diffrho=df_tmp.trialRR[df_tmp.nPostInterval==b].diff()
#             axt.plot(difft.values,diffrho.values,'.',label=str(b)+r' corr='+str(difft.corr(diffrho)))
#         axt.legend(frameon=False)
        
        #return map correlations
#         for b in block_times:
#             difft=df_tmp.tDecision[df_tmp.nPostInterval==b]
#             diffrho=difft[1:].copy()
#             diffrho.index=diffrho.index.values-1
#             difft=difft[:-1]
#             axt.plot(difft.values,diffrho.values,'.',label=str(b)+r' corr='+str(difft.corr(diffrho)))
#         axt.legend(frameon=False)

    ax[1,0].legend(titlestrvec,frameon=False)
    ax[1,1].legend(titlestrvec,frameon=False)
    fig.tight_layout(pad=0.1)
#     fig,ax=pl.subplots(1,1)
#     ax.plot(acorr[1]-acorr[0])
#     ax.set_xlim(1,len(auto_corr)/2)
#     ax.set_xscale('log')
    fig.savefig(save_name+'.png', transparent=True,bbox_inches="tight",dpi=300)

    
def plot_dec_times(df_data,pl,dataset_name,axbase,ax,ymax=None,T=para['T'],start_time=None,itera=0):
    '''
    Computes empirical action policy distributions from a sample ensemble of trials held in the dataframe df_data 
    '''

    if (df_data.tDecision.values==0).sum():
        print('some decisions at t=0!')
        
    ###count distributions of decision events in (N_p,N_m) space 
    dec_counts=np.zeros((T+1,T+1))
    dec_R_counts=np.zeros((T+1,T+1))
    dec_L_counts=np.zeros((T+1,T+1))
    dec_wait_counts=np.zeros((T+1,T+1))
    for it,tdec in enumerate(df_data.tDecision.values):
        curr_traj=df_data.seq.iloc[it]
        #over all trajs 
        dec_counts[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm). int rounds down so should include current state. -1 to map to traj time index,
        #over trajs|action
        if df_data.nChoiceMade.iloc[it]==1:
            dec_R_counts[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm)
        else:# trial_dec[it]==-1:
            dec_L_counts[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm)
        for sit in range(1,int(tdec)): #add wait action up to decision
            dec_wait_counts[np.sum(curr_traj[:sit-1]==1),np.sum(curr_traj[:sit-1]==-1)]+=1 #increment occupancy at given (Np,Nm    
    #add t=0 data
    dec_counts[0,0]=(df_data.tDecision==0).sum()
    dec_R_counts[0,0]=((df_data.tDecision==0) & (df_data.nChoiceMade==1)).sum()
    dec_L_counts[0,0]=((df_data.tDecision==0) & (df_data.nChoiceMade==-1)).sum()
    dec_wait_counts[0,0]=len(df_data)-dec_R_counts[0,0]-dec_L_counts[0,0]
    
    
    mesh=np.meshgrid(range(T+1),range(T+1))
    Npvec=mesh[0].flatten()
    Nmvec=mesh[1].flatten()
    tvec=Npvec+Nmvec
    Nvec=Npvec-Nmvec
    for Np in range(T+1):
        for Nm in range(T+1):
            axbase.scatter(tvec.flatten(),Nvec.flatten(),s=20000*(dec_counts/np.sum(dec_counts)).flatten()**2,marker='+',linewidth=3)
    for Np in range(T+1):
        for Nm in range(T+1):
            if Np+Nm< T+1:
                axbase.scatter(Np+Nm,Np-Nm,s=5,facecolor='w',edgecolor='k',lw=0.5)
    axbase.set_xticks([0,1/3*T,2/3*T,T])
    axbase.set_yticks([-10,0,2/3*T])
    axbase.set_xlim([0,15])
    ###state occupancy count distributions in (N_p,N_m) space
    occupancy_counts=np.zeros((T+1,T+1))
    occupancy_predec_counts=np.zeros((T+1,T+1))
    surv_prob=np.zeros((T+1,T+1))
    Nt_samples=np.cumsum(np.asarray(df_data.seq.tolist()),axis=1) #note that this starts at t=1, so indexing adjusted by -1 below.
    tdec_vec=df_data.tDecision.apply(lambda x:int(x)).values
    for Np in np.arange(T+1):
        for Nm in np.arange(T+1):
            if Np+Nm<=T and Np+Nm>0:#=0
                occupancy_counts[Np,Nm]=np.sum(Nt_samples[:,Np+Nm-1]==Np-Nm)      
                occupancy_predec_counts[Np,Nm]=np.sum(Nt_samples[tdec_vec-1>=Np+Nm-1,Np+Nm-1]==Np-Nm) 
                surv_prob[Np,Nm]=np.sum(tdec_vec[Nt_samples[:,Np+Nm-1]==Np-Nm]>Np+Nm)/occupancy_counts[Np,Nm]
                                          
                #of ones that get here, what fraction have yet to decide
    surv_prob[0,0]=1  
    surv_prob[np.isnan(surv_prob)]=0
    occupancy_counts[0,0]=len(df_data) #all trajs go through (N_p=0,N_m=0).
    occupancy_predec_counts[0,0]=len(df_data)  #all trajs go through (N_p=0,N_m=0).
    
#     dec_dist=np.where(occupancy_predec_counts>0,dec_counts/occupancy_predec_counts,0)#*occupancy_counts/len(df_data)*10
    dec_dist=surv_prob#np.where(occupancy_predec_counts>0,dec_counts/occupancy_predec_counts,0)#*occupancy_counts/len(df_data)*10

    
    ###plot on smoothed coordinates.
    mesh=np.meshgrid(range(T+1),range(T+1))
    Npvec=mesh[0].flatten()
    Nmvec=mesh[1].flatten()
    tvec=Npvec+Nmvec
    Nvec=Npvec-Nmvec
    tvecdense=np.linspace(min(tvec),max(tvec),100)
    Nvecdense=np.linspace(min(Nvec),max(Nvec),100)
    #normalize dist for [0,1] colorscale
    z_d=griddata((tvec,Nvec),dec_dist.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    
    point=8
    offset=0.5
    ax.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.6,0.6,0.6],zorder=5)
    ax.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.6,0.6,0.6],zorder=5)
    for Nm in 2*np.arange(T):
        ax.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=0.5,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',zorder=20)
    for Np in 2*np.arange(T):
        ax.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=0.5,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',zorder=20)
            
    #prob of deciding
    z_d[z_d<0]=0

    CS=ax.contourf(tvecdense,Nvecdense,z_d,30,cmap=pl.cm.twilight,vmax=1., vmin=0.)#,alpha=0.8)
    ax.contour(tvecdense,Nvecdense,z_d,levels=[0.5],linewidths=(1.,),colors=('w'),linestyles=('dotted'),zorder=15)

    ax.set_xlim(0,T)
    ax.set_ylim(-T,T)
    ax.set_xticks([0,1/3*T,2/3*T,T])

    point=8
    offset=0.5
    axbase.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.6,0.6,0.6])
    axbase.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.6,0.6,0.6])
    for Nm in 2*np.arange(T):
        axbase.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=0.5,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for Np in 2*np.arange(T):
        axbase.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=0.5,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
   
    return CS

def plot_policies(df_dict,file_name=None):
    titlestrvec=('subject','model')
    data_type=('act','mod')
    fig,ax=pl.subplots(2,3,figsize=(7,5))
    ax[0,2].set_axis_off()
    ax[1,2].set_axis_off()
    figbase,axbase=pl.subplots(2,2,figsize=(5,5))
    for j,dtype in enumerate(data_type):
        df_tmp=df_dict[dtype]
        block_times=[150,50]
        for pit,post_interval in enumerate(block_times):
            dftmp=df_tmp[df_tmp.nPostInterval==post_interval].reset_index()
            col_bar=plot_dec_times(dftmp,pl,'heuristic_post_'+str(post_interval),axbase[pit,j],ax[pit,j],pit)#,ymax=0.3)
            axbase[pit,j].set_frame_on(False)
            ax[pit,j].set_frame_on(False)
    for axt in (ax,axbase):
        axt[0,0].text(0,10,'slow block')
        axt[1,0].text(0,10,'fast block')
        axt[0,0].set_xticklabels([])
        axt[0,1].set_xticklabels([])
        axt[0,0].set_ylabel('token difference, $N_t$')
        axt[1,0].set_ylabel('token difference, $N_t$')
        axt[1,0].set_xlabel(r'time, $t$')
        axt[1,1].set_xlabel(r'time, $t$')
        axt[0,0].set_title('subject')
        axt[0,1].set_title('subject model')
    fig.subplots_adjust(right=0.8)
    
    fig.tight_layout()
    cbar_ax=fig.add_axes([0.70, 0.15,0.02,0.7])
    cbar_ax.set_title('surv. \n prob.')
    fig.colorbar(col_bar,cax=cbar_ax,ticks=[0, 0.5,1])
    
    figbase.tight_layout()
    if file_name is not None:
       fig.savefig('heuristic_transitions.png', transparent=True,bbox_inches="tight",dpi=300)
       figbase.savefig('heuristic_transitions_base.png', transparent=True,bbox_inches="tight",dpi=300)
    
def plot_transitions(df_dict,ax=None, measure='tDecision'):
    titlestrvec=('subject','model')
    data_type=('act','mod')
    if ax is None:
        fig,ax=pl.subplots(1,1,figsize=(5,5))
    
#     figs,axs=pl.subplots(1,1,figsize=(5,5))
    for j,dtype in enumerate(data_type):
        df_tmp=df_dict[dtype]
        if type(df_tmp) is not list:
            df_tmp=[df_tmp]
        for df_t in df_tmp:
            ax=plot_model_transition(df_t,j,ax,measure=measure)
    #ax.fill_between([-20,0],[ax.get_ylim()[0]]*2,[ax.get_ylim()[1]]*2,color=[0.7]*3,alpha=0.5)
    #ax.legend(frameon=False,ncol=2)#,bbox_to_anchor=(1,1.2))
#     fig.savefig('heuristic_transitions.pdf', transparent=True,bbox_inches="tight",dpi=300)
    return ax

def plot_model_transition(df_tmp,j,ax,measure='tDecision'):
    if measure in df_tmp.columns:
        timevec,data_store=get_transition_ensemble(df_tmp,measure)

        #else:
            #timevec,data_store=get_transition_ensemble(df_tmp,'tDecision')

        #mean &sem
        ls=['.','.']
        colorstr=['gray','C0']
        for pit,data in enumerate(data_store):
            me=np.mean(data,axis=0)
            std_dev=np.std(data,axis=0)
            sem=std_dev/np.sqrt(data.shape[1])
            if j==0:
                ax.plot(timevec,me,ls[j],color=colorstr[pit],zorder=4 if pit==1 else 0,label='data' if pit==1 else None);
    #             ax.plot(ax.get_xlim(),[np.mean(me[timevec<0])]*2,'--',color='C'+str(pit));
    #             ax.plot(ax.get_xlim(),[np.mean(me[-20:])]*2,'--',color='C'+str(pit));
            else:
                ax.fill_between(timevec+1, me - sem, me + sem, color=colorstr[pit],alpha=0.5,label='fit' if pit==1 else None)
        ax.set_xlabel('Trial after context switch')
        ax.set_ylabel('average ')
        ax.legend(frameon=False,loc=5)
        ax.set_xlim(timevec[0],100)
    return ax