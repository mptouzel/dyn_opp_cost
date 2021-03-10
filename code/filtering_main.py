#local external packages
import numpy as np
import time
from functools import partial
#add package to path
import sys,os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
#load local functions
from lib.filter_lib import *

if __name__ == "__main__":
    '''
    A simple script that loads the data from Thura et al. 2016 and runs the model and computes the model error.
    Used to profile using kernprof.py
    ''' 
    
    #load data
    if not os.path.exists(os.getcwd()+'../exp_data/Thura_etal_2016/df_data'):
        df_data=load_data()
        df_data.to_pickle('df_data')
    else:
        df_data=pd.read_pickle('df_data')
    
    #select subset
    subject=1
    df_act=df_data[df_data.idSubject==subject].reset_index(drop=True)
    timevec,data_store_actual=get_transition_ensemble(df_act)
    
    #model paras
    paras=[100,1000] 
    
    #run model
    get_error(paras,df_act,data_store_actual,model='taus_only')
