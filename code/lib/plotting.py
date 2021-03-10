from scipy.interpolate import griddata
from functools import partial
import numpy as np
from scipy.interpolate import griddata

def plot_nocost_dec_times(df_data,pl,dataset_name,axbase,ax,ymax=None,T=15,start_time=None,label_str=None):
    '''
    Computes empirical action policy distributions from a sample ensemble of trials held in the dataframe df_data 
    '''

    if (df_data.tDecision.values==0).sum():
        print('some decisions at t=0!')
        
    ###count distributions of decision events in (N_p,N_m) space 
    dec_counts=np.zeros((T+1,T+1))
    for it,tdec in enumerate(df_data.tDecision.values):
        curr_traj=df_data.seq.iloc[it]
        dec_counts[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm). int rounds down so should include current state. -1 to map to traj time index,  
    #add t=0 data
    dec_counts[0,0]=(df_data.tDecision==0).sum()
    
    #decision events over trial ensemble 
    point=8
    offset=0.5
    axbase.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.8,0.8,0.8])
    axbase.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.8,0.8,0.8])
    for Nm in 2*np.arange(T):
        axbase.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for Np in 2*np.arange(T):
        axbase.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    mesh=np.meshgrid(range(T+1),range(T+1))
    Npvec=mesh[0].flatten()
    Nmvec=mesh[1].flatten()
    tvec=Npvec+Nmvec
    Nvec=Npvec-Nmvec
    axbase.scatter(tvec.flatten(),Nvec.flatten(),s=160000*(dec_counts/np.sum(dec_counts)).flatten()**2,marker='+',linewidth=3,color='b',label=label_str)
    for Np in range(T+1):
        for Nm in range(T+1):
            if Np+Nm< T+1:
                axbase.scatter(Np+Nm,Np-Nm,s=5,facecolor='w',edgecolor='k',lw=0.5)
    axbase.set_xticks([0,1/3*T,2/3*T,T])
    axbase.set_yticks([-10,0,2/3*T])
    axbase.set_xlim([0,15])
    axbase.set_title('trial ensemble')
    axbase.legend(frameon=False)
    
    #state occupancy count distributions in (N_p,N_m) space
    occupancy_predec_counts=np.zeros((T+1,T+1))
    Nt_samples=np.cumsum(np.asarray(df_data.seq.tolist()),axis=1) #note that this starts at t=1, so indexing adjusted by -1 below.
    tdec_vec=df_data.tDecision.apply(lambda x:int(x)).values
    for Np in np.arange(T+1):
        for Nm in np.arange(T+1):
            if Np+Nm<=T and Np+Nm>0:#=0
                occupancy_predec_counts[Np,Nm]=np.sum(Nt_samples[tdec_vec-1>=Np+Nm-1,Np+Nm-1]==Np-Nm) 
    occupancy_predec_counts[0,0]=len(df_data)  #all trajs go through (N_p=0,N_m=0).
    dec_dist=np.where(occupancy_predec_counts>0,dec_counts/occupancy_predec_counts,0)

    #decision events over visitation ensemble
    ax.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.8,0.8,0.8])
    ax.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.8,0.8,0.8])
    for Nm in 2*np.arange(T):
        ax.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for Np in 2*np.arange(T):
        ax.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    ax.scatter(tvec.flatten(),Nvec.flatten(),s=500*dec_dist.flatten()**2,marker='+',linewidth=3,color='r',label=label_str)
    ax.set_xlim(0,T)
    ax.set_ylim(-T,T)
    ax.set_xticks([0,1/3*T,2/3*T,T]) 
    for Np in range(T+1):
        for Nm in range(T+1):
            if Np+Nm< T+1:
                ax.scatter(Np+Nm,Np-Nm,s=5,facecolor='w',edgecolor='k',lw=0.5)
    ax.set_title('visitation ensemble')
    ax.legend(frameon=False)
    #accuracy bias
    in_count=0
    out_count=0
    for Np in range(T+1):
        for Nm in range(T+1):
            if Np>=8 or Nm>=8:
                in_count+=dec_dist[Np,Nm]
            else:
                out_count+=dec_dist[Np,Nm]
                
    return (in_count-out_count)/np.sum(dec_dist)



def plot_dec_times(df_data,pl,dataset_name,axbase,ax,ymax=None,T=15,start_time=None,itera=0):
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
            axbase.scatter(tvec.flatten(),Nvec.flatten(),s=160000*(dec_counts/np.sum(dec_counts)).flatten()**2,marker='+',linewidth=3)
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

#     ###combine to form state frequency distributions
#     dist_tmp=np.zeros(dist.shape)
#     dist_R_tmp=np.zeros(dist_R.shape)
#     dist_L_tmp=np.zeros(dist_L.shape)
#     dist_wait_tmp=np.zeros(dist_L.shape)
#     for Np in range(T+1):
#         for Nm in range(T+1):
#             if occupancy_dist[Np,Nm]>0:
# #             if dist[Np,Nm]>0:
#                 dist_tmp[Np,Nm]=dist[Np,Nm]/occupancy_pre_dist[Np,Nm] if occupancy_pre_dist[Np,Nm]>0 else 0
#                 dist_R_tmp[Np,Nm]=dist_R[Np,Nm]/occupancy_dist[Np,Nm] 
#                 dist_L_tmp[Np,Nm]=dist_L[Np,Nm]/occupancy_dist[Np,Nm] 
#                 dist_wait_tmp[Np,Nm]=1.-dist_R_tmp[Np,Nm] - dist_L_tmp[Np,Nm]#dist_wait[Np,Nm]/occupancy_dist[Np,Nm]   
# #                 dist_R_tmp[Np,Nm]=dist_R[Np,Nm]/dist[Np,Nm] 
# #                 dist_L_tmp[Np,Nm]=dist_L[Np,Nm]/dist[Np,Nm] 
# #                 dist_wait_tmp[Np,Nm]= dist_wait[Np,Nm]/dist[Np,Nm] 
#     dist=dist_tmp
#     dist_R=dist_R_tmp
#     dist_L=dist_L_tmp
#     dist_wait=dist_wait_tmp

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

#     z_R=griddata((tvec,Nvec),dist_R.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
#     z_L=griddata((tvec,Nvec),dist_L.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
#     z_wait=griddata((tvec,Nvec),dist_wait.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
# #     z_R=griddata((tvec,Nvec),dist_R.flatten()/np.sum(np.sum(dist_R)), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
# #     z_L=griddata((tvec,Nvec),dist_L.flatten()/np.sum(np.sum(dist_L)), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
# #     z_wait=griddata((tvec,Nvec),dist_wait.flatten()/np.sum(np.sum(dist_wait)), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
#     print(dist_wait[0,0])
#     fig, ax=pl.subplots(1,4,frameon=False,figsize=(26,7))
#     strp=['left','right','wait']
#     zvec=[z_R,z_L,z_wait] #wrong order here, because something above is in wrong order...
#     for p,z in enumerate(zvec):
#         z[z<0]=0
#         z[z>1]=1
#         CS=ax[p].contourf(tvecdense,Nvecdense,z,30,cmap=pl.cm.jet,vmax=1., vmin=0.)
#         if p==0:
#             ax[p].set_ylabel(r'token difference, $N_t$')
#         else:
#             ax[p].set_yticklabels('')
#         ax[p].set_xlim(0,T)
#         ax[p].set_ylim(-T,T)
#         ax[p].set_xticks([0,1/3*T,2/3*T,T])
#         ax[p].set_xlabel(r'time, $t$')
#         ax[p].set_title(r'$a=\textrm{'+strp[p]+'}$')
#         for Np in range(T+1):
#             for Nm in range(T+1):
#                 if Np+Nm< T+1:
#                     ax[p].scatter(Np+Nm,Np-Nm,s=10,facecolor='w',edgecolor='k',lw=0.5)
#     ax[2].figure.colorbar(CS)
#     #response time distirbutions
#     df_data.tDecision.hist(ax=ax[3],bins=np.arange(T+2)-0.5,density=True)
# #     t_dist=np.zeros((T+1,))
# #     for t in range(T+1):
# #         t_dist[t]=np.diagonal(np.flipud(dist),offset=-(T)+t).mean() #marginalize over state
# #     ax[3].bar(range(T+1),t_dist)
#     ax[3].set_ylabel('frequency')
#     ax[3].set_xlabel(r'time, $t$')
#     ax[3].set_xlim(0,T)
#     ax[3].set_xticks([0,1/3*T,2/3*T,T])
#     ax[3].set_title('commitment time histogram')
#     ax[3].set_ylim(0,ymax)

#     fig.tight_layout()
#     fig.suptitle(r'empirical primate policy, $\pi(a|s=(N_t,t))$',y=1.05)
#     fig.savefig('primate_policy_poster_'+dataset_name+'.pdf', transparent=True,bbox_inches="tight",dpi=300)

    point=8
    offset=0.5
    ax.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.6,0.6,0.6],zorder=5)
    ax.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.6,0.6,0.6],zorder=5)
    for Nm in 2*np.arange(T):
        ax.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',zorder=20)
    for Np in 2*np.arange(T):
        ax.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',zorder=20)
#     for Np in range(T+1):
#         for Nm in range(T+1):
#             ax.scatter(tvec.flatten(),Nvec.flatten(),s=500*dec_dist.flatten()**2,marker='+',linewidth=3)
            
    #prob of deciding
#     fig, ax=pl.subplots(1,1,figsize=(6,4))#frameon=False,
    z_d[z_d<0]=0
#     z_d[z_d>1]=1
#     z_d[np.isnan(z_d)]=-np.Inf
#     print(np.sum((np.isnan(z_d))))
#     print(z_d)
    CS=ax.contourf(tvecdense,Nvecdense,z_d,30,cmap=pl.cm.twilight,vmax=1., vmin=0.)#,alpha=0.8)
#     CS=ax.contourf(tvecdense,Nvecdense,z_d,30,zorder=2)
    ax.contour(tvecdense,Nvecdense,z_d,levels=[0.5],linewidths=(1.5,),colors=('w'),linestyles=('dotted'),zorder=15)
#     ax.grid()
#     ax.figure.colorbar(CS)
#     ax.set_ylabel(r'token difference, $N_t$')
#     ax.set_yticklabels('')
    ax.set_xlim(0,T)
    ax.set_ylim(-T,T)
    ax.set_xticks([0,1/3*T,2/3*T,T])
#     ax.set_xlabel(r'time, $t$')
#     ax.set_title(r'$\textrm{Pr}(\textrm{decide}|N_t,t)$')
#     ax.set_title(r'$t='+str(start_time)+'$')
#     ax.axis('off')
    point=8
    offset=0.5
    axbase.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.6,0.6,0.6])
    axbase.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.6,0.6,0.6])
#     for Np in range(T+1):
#         for Nm in range(T+1):
#             if Np+Nm< T+1:
#                 ax.scatter(Np+Nm,Np-Nm,s=5,facecolor='w',edgecolor='k',lw=0.5)

#     axbase.imshow(dec_counts[:8,:8],origin='lower')  
    for Nm in 2*np.arange(T):
        axbase.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    for Np in 2*np.arange(T):
        axbase.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
#     fig.savefig('primate_policy_poster_P_dec_'+dataset_name+'.pdf', transparent=True,bbox_inches="tight",dpi=300)

    

#     fig, ax=pl.subplots(1,1,frameon=False,figsize=(6,4))
#     CS=ax.contourf(tvecdense,Nvecdense,(z_L+z_R)/2,30,cmap=pl.cm.jet)
#     ax.set_ylabel(r'token difference, $N_t$')
# #     ax.set_yticklabels('')
#     ax.set_xlim(0,T)
#     ax.set_ylim(-T,T)
#     ax.set_xticks([0,1/3*T,2/3*T,T])
#     ax.set_xlabel(r'time, $t$')
#     ax.set_title(r'$\textrm{Pr}(\textrm{decide}|N_t,t)$')
# #     ax.set_title(r'$t='+str(start_time)+'$')
# #     ax.axis('off')

#     for Np in range(T+1):
#         for Nm in range(T+1):
#             if Np+Nm< T+1:
#                 ax.scatter(Np+Nm,Np-Nm,s=10,facecolor='w',edgecolor='k',lw=0.5)
#     ax.figure.colorbar(CS)

     #accuracy bias
    in_count=0
    out_count=0
    for Np in range(T+1):
        for Nm in range(T+1):
            if Np>=8 or Nm>=8:
                in_count+=dec_dist[Np,Nm]
            else:
                out_count+=dec_dist[Np,Nm]
                
    return CS#(in_count-out_count)/np.sum(dec_dist)


def plot_dec_times(df_data,pl,dataset_name,ymax=1,T=15,ax=None,savefigs=False):
    '''
    Computes empirical action policy distributions from a sample ensemble of trials held in the dataframe df_data 
    '''

    if (df_data.tDecision.values==0).sum():
        print(str((df_data.tDecision.values==0).sum())+' decisions at t=0!')
    if (df_data.tDecision.values<0).sum():
        print(str((df_data.tDecision.values<0).sum())+' decisions before t=0!')

    ###count distributions of decision events in (N_p,N_m) space 
    dist=np.zeros((T+1,T+1))
    dist_R=np.zeros((T+1,T+1))
    dist_L=np.zeros((T+1,T+1))
    dist_wait=np.zeros((T+1,T+1))
    for it,tdec in enumerate(df_data.tDecision.values):
        curr_traj=df_data.seq.iloc[it]

        #indexing logic: 
        #-int rounds tdec down to last token jump time, t, (i.e. current evidence), 
        #-curr_traj starts at t=0 so curr_traj[:t] gives t token jumps
        
        #over all trajs 
        dist[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm). int rounds down so should include current state. -1 to map to traj time index,
        #over trajs|action
        if df_data.nChoiceMade.iloc[it]==1:
            dist_R[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm)
        else:# df_data.nChoiceMade.iloc[it]==-1:
            dist_L[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm)
        for sit in range(1,int(tdec)): #add wait action up to decision
            dist_wait[np.sum(curr_traj[:sit-1]==1),np.sum(curr_traj[:sit-1]==-1)]+=1 #increment occupancy at given (Np,Nm)

    dist_corr=np.zeros((T+1,T+1))
    for it,tdec in enumerate(df_data.tDecision.values):
        curr_traj=df_data.seq.iloc[it]
        #over all trajs 
        dist_corr[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=(df_data.nChoiceMade.iloc[it]==df_data.nCorrectChoice.iloc[it])

    #add t=0 data
    dist[0,0]=(df_data.tDecision==0).sum()
    dist_R[0,0]=((df_data.tDecision==0) & (df_data.nChoiceMade==1)).sum()
    dist_L[0,0]=((df_data.tDecision==0) & (df_data.nChoiceMade==-1)).sum()
    dist_wait[0,0]=len(df_data)-dist_R[0,0]-dist_L[0,0]
    print(dist_wait[0,0])

    ###state occupancy count distributions in (N_p,N_m) space
    occupancy_dist=np.zeros((T+1,T+1))
    Nt_samples=np.cumsum(np.asarray(df_data.seq.tolist()),axis=1) #note that this starts at t=1, so indexing adjusted by -1 below.
    tdec_vec=dfb.tDecision.values
    for Np in np.arange(T+1):
        for Nm in np.arange(T+1):
            if Np+Nm<=T and Np+Nm>0:#=0
                occupancy_dist[Np,Nm]=np.sum(Nt_samples[tdec_vec>=Np+Nm-1,Np+Nm-1]==Np-Nm)     
    occupancy_dist[0,0]=len(df_data) #all trajs go through (N_p=0,N_m=0).

    ###combine to form state frequency distributions
    dist_tmp=np.zeros(dist.shape)
    dist_R_tmp=np.zeros(dist_R.shape)
    dist_L_tmp=np.zeros(dist_L.shape)
    dist_wait_tmp=np.zeros(dist_L.shape)
    dist_corr_tmp=np.zeros(dist_L.shape)
    for Np in range(T+1):
        for Nm in range(T+1):
            if occupancy_dist[Np,Nm]>0:
    #             if dist[Np,Nm]>0:
                dist_tmp[Np,Nm]=dist[Np,Nm]/occupancy_dist[Np,Nm]
                dist_R_tmp[Np,Nm]=dist_R[Np,Nm]/occupancy_dist[Np,Nm] 
                dist_L_tmp[Np,Nm]=dist_L[Np,Nm]/occupancy_dist[Np,Nm] 
                dist_wait_tmp[Np,Nm]=1.-dist_R_tmp[Np,Nm] - dist_L_tmp[Np,Nm]#dist_wait[Np,Nm]/occupancy_dist[Np,Nm]   
                dist_corr_tmp[Np,Nm]=dist_corr[Np,Nm]/dist[Np,Nm] 

                #                 dist_R_tmp[Np,Nm]=dist_R[Np,Nm]/dist[Np,Nm] 
    #                 dist_L_tmp[Np,Nm]=dist_L[Np,Nm]/dist[Np,Nm] 
    #                 dist_wait_tmp[Np,Nm]= dist_wait[Np,Nm]/dist[Np,Nm] 
    dist=dist_tmp
    dist_R=dist_R_tmp
    dist_L=dist_L_tmp
    dist_wait=dist_wait_tmp
    dist_corr=dist_corr_tmp

    ###plot on smoothed coordinates.
    mesh=np.meshgrid(range(T+1),range(T+1))
    Npvec=mesh[0].flatten()
    Nmvec=mesh[1].flatten()
    tvec=Npvec+Nmvec
    Nvec=Npvec-Nmvec
    tvecdense=np.linspace(min(tvec),max(tvec),100)
    Nvecdense=np.linspace(min(Nvec),max(Nvec),100)
    #normalize dist for [0,1] colorscale
    z_d=griddata((tvec,Nvec),dist.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')

    z_R=griddata((tvec,Nvec),dist_R.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    z_L=griddata((tvec,Nvec),dist_L.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    z_wait=griddata((tvec,Nvec),dist_wait.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    #     z_R=griddata((tvec,Nvec),dist_R.flatten()/np.sum(np.sum(dist_R)), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    #     z_L=griddata((tvec,Nvec),dist_L.flatten()/np.sum(np.sum(dist_L)), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    #     z_wait=griddata((tvec,Nvec),dist_wait.flatten()/np.sum(np.sum(dist_wait)), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')


    print(dist_wait[0,0])
    fig, ax=pl.subplots(1,4,frameon=False,figsize=(26,7))
    strp=['left','right','wait']
    zvec=[z_R,z_L,z_wait] #wrong order here, because something above is in wrong order...
    for p,z in enumerate(zvec):
        z[z<0]=0
        z[z>1]=1
        CS=ax[p].contourf(tvecdense,Nvecdense,z,30,cmap=pl.cm.jet,vmax=1., vmin=0.)
        if p==0:
            ax[p].set_ylabel(r'token difference, $N_t$')
        else:
            ax[p].set_yticklabels('')
        ax[p].set_xlim(0,T)
        ax[p].set_ylim(-T,T)
        ax[p].set_xticks([0,1/3*T,2/3*T,T])
        ax[p].set_xlabel(r'time, $t$')
        ax[p].set_title(r'$a=\textrm{'+strp[p]+'}$')
        for Np in range(T+1):
            for Nm in range(T+1):
                if Np+Nm< T+1:
                    ax[p].scatter(Np+Nm,Np-Nm,s=10,facecolor='w',edgecolor='k',lw=0.5)
    ax[2].figure.colorbar(CS)
    #response time distirbutions
    df_data.tDecision.hist(ax=ax[3],bins=np.arange(T+2)-0.5,density=True)
#     t_dist=np.zeros((T+1,))
#     for t in range(T+1):
#         t_dist[t]=np.diagonal(np.flipud(dist),offset=-(T)+t).mean() #marginalize over state
#     ax[3].bar(range(T+1),t_dist)
    ax[3].set_ylabel('frequency')
    ax[3].set_xlabel(r'time, $t$')
    ax[3].set_xlim(0,T)
    ax[3].set_xticks([0,1/3*T,2/3*T,T])
    ax[3].set_title('commitment time histogram')
    ax[3].set_ylim(0,ymax)

    fig.tight_layout()
    fig.suptitle(r'empirical primate policy, $\pi(a|s=(N_t,t))$',y=1.05)
    if savefigs:
        fig.savefig('primate_policy_poster_'+dataset_name+'.pdf', transparent=True,bbox_inches="tight",dpi=300)

    #prob of deciding
    fig, ax=pl.subplots(1,1,frameon=False,figsize=(3,2))
    z_d[z_d<0]=0
    CS=ax.contourf(tvecdense,Nvecdense,z_d,30,cmap=pl.cm.jet)
    ax.set_ylabel(r'token difference, $N_t$')
    ax.set_yticklabels('')
    ax.set_xlim(0,T)
    ax.set_ylim(-T,T)
    ax.set_xticks([0,1/3*T,2/3*T,T])
    ax.set_xlabel(r'time, $t$')
    ax.set_title(r'$\textrm{Pr}(\textrm{decide}|N_t,t)$')
    ax.axis('off')
    for Np in range(T+1):
        for Nm in range(T+1):
            if Np+Nm< T+1:
                ax.scatter(Np+Nm,Np-Nm,s=10,facecolor='w',edgecolor='k',lw=0.5)
    ax.figure.colorbar(CS)
    if savefigs:
        fig.savefig('primate_policy_poster_P_dec_'+dataset_name+'.pdf', transparent=True,bbox_inches="tight",dpi=300)

#         z_corr=griddata((tvec,Nvec),dist_corr.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
#         fig, ax=pl.subplots(1,1,frameon=False,figsize=(3,2))
#     #     z_corr[z_d<0]=0
#         CS=ax.contourf(tvecdense,Nvecdense,z_corr,30,cmap=pl.cm.jet)
#         ax.set_ylabel(r'token difference, $N_t$')
#         ax.set_yticklabels('')
#         ax.set_xlim(0,T)
#         ax.set_ylim(-T,T)
#         ax.set_xticks([0,1/3*T,2/3*T,T])
#         ax.set_xlabel(r'time, $t$')
#         ax.set_title(r'$\textrm{Pr}(\textrm{correct}|N_t,t)$')
#         ax.axis('off')
#         for Np in range(T+1):
#             for Nm in range(T+1):
#                 if Np+Nm< T+1:
#                     ax.scatter(Np+Nm,Np-Nm,s=10,facecolor='w',edgecolor='k',lw=0.5)
#         ax.figure.colorbar(CS)

    
    
  ##the 2D histogram plots  
#     from mpl_toolkits.mplot3d import Axes3D
#     fig=pl.figure()

#     ax=fig.add_subplot(111,projection='3d')
#     x,y=np.meshgrid(range(T+1),range(T+1))
#     dist=dist.flatten()
#     ax.bar3d(x.flatten(),y.flatten(),np.zeros(dist.shape),np.ones(dist.shape),np.ones(dist.shape),dist)


 ## the Pcorr belief space plot
    #for Nm in np.arange(T+1):
    #    ax.plot(np.arange(T+1-Nm)+Nm,dist_corr[:T+1-Nm,Nm],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    #for Np in np.arange(T+1):
    #    ax.plot(np.arange(T+1-Np)+Np,dist_corr[Np,:T+1-Np],'ko-',ms=3,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')

#     error=np.sqrt(dist_corr*(1-dist_corr)/dist)
#     for Nm in np.arange(T+1):
#         if Nm==0:
#             p=ax.errorbar(np.arange(T+1-Nm)+Nm+ymax,dist_corr[:T+1-Nm,Nm],yerr=error[:T+1-Nm,Nm],fmt='o',ms=3,lw=0.5,mew=0.5)
#             col=p[-1][0].get_color()[0]
#             print(p[-1][0].get_color())
#         else:
#             ax.errorbar(np.arange(T+1-Nm)+Nm+ymax,dist_corr[:T+1-Nm,Nm],yerr=error[:T+1-Nm,Nm],fmt='o',ms=3,lw=0.5,color=col,mew=0.5)

#     for Np in np.arange(T+1):
#         ax.errorbar(np.arange(T+1-Np)+Np+ymax,dist_corr[Np,:T+1-Np],yerr=error[Np,:T+1-Np],fmt='o',ms=3,lw=0.5,color=col,mew=0.5)

#         np.save('primate_p_success_'+dataset_name+'.npy',dist_corr)
#         np.save('primate_p_success_'+dataset_name+'_num.npy',dist)

from scipy.interpolate import griddata
def plot_dec_times(df_data,pl,dataset_name,axbase,ax,ymax=None,T=15,start_time=None):
    '''
    Computes empirical action policy distributions from a sample ensemble of trials held in the dataframe df_data 
    '''

    if (df_data.tDecision.values==0).sum():
        print('some decisions at t=0!')
        
    ###count distributions of decision events in (N_p,N_m) space 
    dec_counts=np.zeros((T+1,T+1))
    for it,tdec in enumerate(df_data.tDecision.values):
        curr_traj=df_data.seq.iloc[it]
        dec_counts[np.sum(curr_traj[:int(tdec)]==1),np.sum(curr_traj[:int(tdec)]==-1)]+=1 #increment occupancy at given (Np,Nm). int rounds down so should include current state. -1 to map to traj time index,  
    #add t=0 data
    dec_counts[0,0]=(df_data.tDecision==0).sum()
    
    point=8
    offset=0.5
#     axbase.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.8,0.8,0.8])
#     axbase.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.8,0.8,0.8])
#     for Nm in 2*np.arange(T):
#         axbase.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
#     for Np in 2*np.arange(T):
#         axbase.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
    mesh=np.meshgrid(range(T+1),range(T+1))
    Npvec=mesh[0].flatten()
    Nmvec=mesh[1].flatten()
    tvec=Npvec+Nmvec
    Nvec=Npvec-Nmvec
#     axbase.scatter(tvec.flatten(),Nvec.flatten(),s=160000*(dec_counts/np.sum(dec_counts)).flatten()**2,marker='+',linewidth=3)
#     for Np in range(T+1):
#         for Nm in range(T+1):
#             if Np+Nm< T+1:
#                 axbase.scatter(Np+Nm,Np-Nm,s=5,facecolor='w',edgecolor='k',lw=0.5)
#     axbase.set_xticks([0,1/3*T,2/3*T,T])
#     axbase.set_yticks([-10,0,2/3*T])
#     axbase.set_xlim([0,15])
    
    #state occupancy count distributions in (N_p,N_m) space
    occupancy_predec_counts=np.zeros((T+1,T+1))
    occupancy_counts=np.zeros((T+1,T+1))
    Nt_samples=np.cumsum(np.asarray(df_data.seq.tolist()),axis=1) #note that this starts at t=1, so indexing adjusted by -1 below.
    tdec_vec=df_data.tDecision.apply(lambda x:int(x)).values
    surv_prob=np.zeros((T+1,T+1))
    for Np in np.arange(T+1):
        for Nm in np.arange(T+1):
            if Np+Nm<=T and Np+Nm>0:#=0
                occupancy_counts[Np,Nm]=np.sum(Nt_samples[:,Np+Nm-1]==Np-Nm) 
                occupancy_predec_counts[Np,Nm]=np.sum(Nt_samples[tdec_vec-1>=Np+Nm-1,Np+Nm-1]==Np-Nm) 
                surv_prob[Np,Nm]=np.sum(tdec_vec[Nt_samples[:,Np+Nm-1]==Np-Nm]>Np+Nm)/occupancy_counts[Np,Nm]
    occupancy_predec_counts[0,0]=len(df_data)  #all trajs go through (N_p=0,N_m=0).
    dec_dist=np.where(occupancy_predec_counts>0,dec_counts/occupancy_predec_counts,0)
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
    #normalize dist for [0,1] colorscale
    z_d=griddata((tvec,Nvec),surv_prob.flatten(), (tvecdense[None,:],Nvecdense[:,None]),method='cubic')
    point=8
    offset=0.5
    ax.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.6,0.6,0.6],zorder=5)
    ax.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.6,0.6,0.6],zorder=5)
    for Nm in 2*np.arange(T):
        ax.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',zorder=20)
    for Np in 2*np.arange(T):
        ax.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k',zorder=20)
    z_d[z_d<0]=0
    CS=ax.contourf(tvecdense,Nvecdense,z_d,30,cmap=pl.cm.twilight,vmax=1., vmin=0.)#,alpha=0.8)
    ax.contour(tvecdense,Nvecdense,z_d,levels=[0.5],linewidths=(1.5,),colors=('w'),linestyles=('dotted'),zorder=15)
    ax.set_xlim(0,T)
    ax.set_ylim(-T,T)
    ax.set_xticks([0,1/3*T,2/3*T,T])


#     ax.fill_between(-offset+np.arange(point,2*point+1),-offset+point-np.arange(point+1),-offset+point+np.arange(point+1),color=[0.8,0.8,0.8])
#     ax.fill_between(-offset+np.arange(point,2*point+1),offset-point-np.arange(point+1),offset-point+np.arange(point+1),color=[0.8,0.8,0.8])
#     for Nm in 2*np.arange(T):
#         ax.plot(np.arange(T+1-Nm/2)+Nm/2,np.arange(T+1-Nm/2)-Nm/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')
#     for Np in 2*np.arange(T):
#         ax.plot(np.arange(T+1-Np/2)+Np/2,np.arange(T+1-Np/2)-Np/2,'ko',ms=1,lw=0.5,color='k',mew=0.5,mfc='k',mec='k')

#     ax.scatter(tvec.flatten(),Nvec.flatten(),s=500*surv_prob.flatten()**2,marker='+',linewidth=3)
#     ax.set_xlim(0,T)
#     ax.set_ylim(-T,T)
#     ax.set_xticks([0,1/3*T,2/3*T,T]) 
#     for Np in range(T+1):
#         for Nm in range(T+1):
#             if Np+Nm< T+1:
#                 ax.scatter(Np+Nm,Np-Nm,s=5,facecolor='w',edgecolor='k',lw=0.5)

    #accuracy bias
    in_count=0
    out_count=0
    for Np in range(T+1):
        for Nm in range(T+1):
            if Np>=8 or Nm>=8:
                in_count+=dec_dist[Np,Nm]
            else:
                out_count+=dec_dist[Np,Nm]
                
    return (in_count-out_count)/np.sum(dec_dist)

# def add_subplot_axes(ax,rect,axisbg='w'):
#     fig = pl.gcf()
#     box = ax.get_position()
#     width = box.width
#     height = box.height
#     inax_position  = ax.transAxes.transform(rect[0:2])
#     transFigure = fig.transFigure.inverted()
#     infig_position = transFigure.transform(inax_position)
#     x = infig_position[0]
#     y = infig_position[1]
#     width *= rect[2]
#     height *= rect[3]  # <= Typo was here
#     subax = fig.add_axes([x,y,width,height],facecolor=axisbg)
#     x_labelsize = subax.get_xticklabels()[0].get_size()
#     y_labelsize = subax.get_yticklabels()[0].get_size()
#     x_labelsize *= rect[2]**0.5
#     y_labelsize *= rect[3]**0.5
#     subax.xaxis.set_tick_params(labelsize=x_labelsize)
#     subax.yaxis.set_tick_params(labelsize=y_labelsize)
#     return subax
