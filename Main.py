from numpy import matlib
from BIO import BIO
from DSOA import DSOA
from Global_vars import Global_vars
from Model_DQN import Model_DQN
from Objective_Function import Obj_fun_RL
from Plot_Results import plot_Results, plot_Reward, plot_Fitness
from Proposed import Proposed
from RPO import RPO
from WSA import WSA
from structtype import structtype
import numpy as np
from dqn_Parameter import dqn_Parameter
from env_hypers import env_hypers


#  Initialization
an = 0
if an == 1:
    # df = pd.read_pickle('df.pkl')
    Var = [structtype() for n in range(5)]
    No_of_Server = [100, 120, 140, 160, 180]
    No_of_VM = [500, 600, 700, 800, 900]
    no_of_tasks = [50, 100, 150, 200, 250]
    Pmax = 215  # maximum power(W)
    VC_limit = [1, 98]  # CPU
    VM_limit = [0, 80]  # GB Ram
    PC_Value = 500  # CPU
    PM_Value = 500  # GB Ram
    for i in range(len(no_of_tasks)):
        VC = np.random.randint(low=VC_limit[0], high=VC_limit[1], size=[1, No_of_VM[i]])  # VM - CPU
        VM = np.random.randint(low=VM_limit[0], high=VM_limit[1], size=[1, No_of_VM[i]])  # VM - Memory
        PC = PC_Value * np.ones(No_of_Server[i])  # Server - CPU
        PM = PM_Value * np.ones(No_of_Server[i])  # Server - Memory
        BW = np.random.randint(5, 50, size=No_of_VM[i])  # VM - CPU
        Var[i].Active_Servers = np.random.randint(No_of_Server[i], size=[No_of_VM[i]])  # VM - CPU
        Var[i].No_of_Server = No_of_Server[i]
        Var[i].No_of_VM = No_of_VM[i]
        Var[i].no_of_tasks = no_of_tasks[i]
        Var[i].VC_limit = VC_limit
        Var[i].VM_limit = VM_limit
        Var[i].PC_Value = PC_Value
        Var[i].PM_Value = PM_Value
        Var[i].VC = VC
        Var[i].VM = VM
        Var[i].PC = PC
        Var[i].PM = PM
        Var[i].Pmax = Pmax
        Var[i].BW = BW
    np.save('Var.npy', Var)

# Optimization for Task Allocation
an = 0
if an == 1:
    Bestsol = []
    Fitness = []
    Var = np.load('Var.npy', allow_pickle=True)

    for n in range(len(Var)):
        Info = Var[n]
        C = Var[n].Active_Servers  # active  server  after placement and used for migration
        Npop = 10
        Chlen = Var[n].no_of_tasks + 1 # No.of Task
        xmin = matlib.repmat([np.ones((Npop, Chlen-1))], Npop, 100)
        xmax = matlib.repmat([np.ones((Npop, Chlen-1))* Var[n].no_of_tasks], Npop, 1000)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = Obj_fun_RL
        max_iter = 250

        Global_vars.Chlen = Chlen
        Global_vars.Info = Var[n]
        Global_vars.C = C

        print('RPO....')
        [bestfit1, fitness1, bestsol1, Time1] = RPO(initsol, fname, xmin, xmax, max_iter)

        print('DSOA....')
        [bestfit2, fitness2, bestsol2, Time2] = DSOA(initsol, fname, xmin, xmax, max_iter)

        print('WSA....')
        [bestfit3, fitness3, bestsol3, Time3] = WSA(initsol, fname, xmin, xmax, max_iter)

        print('BIO....')
        [bestfit4, fitness4, bestsol4, Time4] = BIO(initsol, fname, xmin, xmax, max_iter)

        print('Proposed....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, max_iter)

        Bestsol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        Fitness.append([fitness1, fitness2, fitness3, fitness4, fitness5])
    np.save('Bestsol.npy', Bestsol)
    np.save('Fitness.npy', Fitness)

# Reinforcement Learning
an = 0
if an == 1:
    TDMA_Rewards = []
    Agent_Rewards = []
    No_of_Task =[50, 100, 150, 200, 250]
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    for n in range(len(No_of_Task)):
        Episode_Reward = []
        Maximum_Reward = []
        env_hypers.no_of_tasks = No_of_Task[n]
        Bestsol = BestSol[n]
        for i in range(Bestsol.shape[0]):
            bestsol = Bestsol[i]
            dqn_Parameter.memory_size = bestsol[0]  # learning rate for actor
            dqn_Parameter.replace_target_iter = bestsol[1]  # learning rate for critic
            dqn_Parameter.batch_size = bestsol[2]  # reward discount
            dqn_Parameter.learning_rate = bestsol[3]  # soft replacement
            sol = np.reshape(bestsol[4:], (10, len(bestsol[4:]) // 10))
            max_iter = sol[1]
            agent_reward_list, TDMA_reward_list, Total_reward = Model_DQN(max_iter, sol)
            Episode_Reward.append(agent_reward_list)
            Maximum_Reward.append(TDMA_reward_list)
        TDMA_Rewards.append(Episode_Reward)
        Agent_Rewards.append(Maximum_Reward)
    np.save('Reward.npy', Agent_Rewards)

plot_Results()
plot_Fitness()
plot_Reward()