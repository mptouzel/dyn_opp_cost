import numpy as np

T=11

def generate_states(T=11):
    state_dict = {}
    counter=0
    for i in range(T+1):
        for j in range(T-i+1):
            state_dict[counter] = [i,j]
            counter += 1
    return state_dict

state_dict = generate_states(T=T)


def get_neighnours(s_index):
    s = state_dict[s_index]
    if sum(s) < T:
        neighbours = [[s[0]+1, s[1]], [s[0], s[1]+1]]
        n_indice = [list(state_dict.keys())[list(state_dict.values()).index(n)] for n in neighbours]
        # n = [s_index + 1, s_index + T]
        return n_indice
    else:
        return None

def get_transition_probablity_with_action(s_index, s_prime_index, a=None):
    ns = get_neighnours(s_index)

    if a == 'wait':
        if ns is not None and s_prime_index in ns:
            return 1/2
        elif ns is None:
            if s_prime_index == 0:
                return 1
            else:
                return 0
        else:
            return 0

    if a == 'right' or a == 'left':
        if s_prime_index == 0:
            return 1
        else:
            return 0



from scipy.special import binom

def get_probabilistic_transition_reward(s, a=None, r_correct=1, r_incorrect=-1):
    t = sum(s)
    Nt = s[1] - s[0]
    Nt_plus=(t+Nt)/2.
    lower_bound=np.ceil(T/2-Nt_plus)
    p_plus = 0
    p_minus = 0
    if lower_bound>0:
        kvec=np.arange(lower_bound,T-t+1,dtype=int)
        p_plus = np.power(0.5,T-t)*np.sum( [binom(T - t,k) for k in kvec])
    else:
        p_plus = 1

    p_minus = 1 - p_plus


    # You get a reward only if you choose an action
    # if you are in a terminal state and you do nothing you get 0, if you decide when you are at a terminal state you get a rewad of 1 or minus -1 according to your choice
    if a == 'right':
        return p_plus*r_correct+(1-p_plus)*r_incorrect
    elif a =='left':
        return p_minus*r_correct+(1-p_minus)*r_incorrect
    else:
        return 0


actions = ['wait', 'left','right']

import mdptoolbox as mdp
import pickle

if __name__ == '__main__':

    rvi_data = {}
    vi_data = {}


    r_incorrect_space = np.linspace(-2,0,41)

    state_array = list(state_dict.keys())
    state_array = np.array(state_array)
    state_size = state_array.size

    P = np.zeros((len(actions), state_size, state_size))

    for a in range(len(actions)):
        for i in range(state_size):
            for j in range(state_size):
                P[a,i,j] = get_transition_probablity_with_action(i,j,a=actions[a])

    for i,r_inc in enumerate(r_incorrect_space):

        R = np.zeros((state_size,len(actions)))

        for s_index in range(state_size):
                for a in range(len(actions)):
                    R[s_index,a] = get_probabilistic_transition_reward(s=state_dict[s_index],a=actions[a], r_correct=1, r_incorrect=r_inc)


        rvi = mdp.mdp.RelativeValueIteration(P, R, epsilon=1e-8, max_iter=1e6)
        rvi.run()

        rvi_data[i] = {}
        rvi_data[i]['r_inc'] = r_inc
        rvi_data[i]['avg_r'] = rvi.average_reward
        rvi_policy = np.array(rvi.policy)
        rvi_policy = rvi_policy.reshape(-1,state_size)
        rvi_data[i]['rvi_policy'] = rvi_policy
        # print(rvi.average_reward)
        rvi_data[i]['rvi_iter'] = rvi.iter
        # print(rvi.iter)
        # print(rvi.error)
        rvi_data[i]['rvi_error'] = rvi.error

        vi_data[i] = {}
        

        for gamma in [0.01, 0.4, 0.6, 0.8, 0.9, 0.99]:
            vi_data[i][gamma] = {}
            vi = mdp.mdp.ValueIteration(P, R, discount=gamma, epsilon=1e-8, max_iter=1e6)
            vi.run()
            vi_data[i][gamma]['r_inc'] = r_inc
            vi_data[i][gamma]['V'] = vi.V
            vi_data[i][gamma]['vi_policy'] = np.array(vi.policy)
            vi_data[i][gamma]['vi_iter'] = vi.iter
            vi_data[i][gamma]['vi_error'] = vi.error
        

    with open('rvi_data.pickle', 'wb') as handle:
        pickle.dump(rvi_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('vi_data.pickle', 'wb') as handle:
        pickle.dump(vi_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
