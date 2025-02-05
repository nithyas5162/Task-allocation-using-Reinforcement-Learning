import numpy as np
from Global_vars import Global_vars
import scipy.integrate as spi
import time
from dqn_Parameter import dqn_Parameter
from Model_DQN import Model_DQN


def Obj_fun_RL(Soln):
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i]
            dqn_Parameter.memory_size = int(sol[0])  # learning rate for actor
            dqn_Parameter.replace_target_iter = int(sol[1])  # learning rate for critic
            dqn_Parameter.batch_size = int(sol[2])  # reward discount
            dqn_Parameter.learning_rate = int(sol[3])  # soft replacement
            max_iter = 100
            agent_reward_list, TDMA_reward_list, Total_reward = Model_DQN(max_iter, sol)
            throughput = Throughput(agent_reward_list, TDMA_reward_list)
            Fitn[i] = 1 / np.max(throughput)
        return Fitn
    else:
        sol = Soln
        dqn_Parameter.memory_size = int(sol[0])  # learning rate for actor
        dqn_Parameter.replace_target_iter = sol[1]  # learning rate for critic
        dqn_Parameter.batch_size = sol[2]  # reward discount
        dqn_Parameter.learning_rate = sol[3]  # soft replacement
        max_iter = 100
        agent_reward_list, TDMA_reward_list, Total_reward = Model_DQN(max_iter, sol)
        throughput = Throughput(agent_reward_list, TDMA_reward_list)
        Fitn = 1 / np.max(throughput)
    return Fitn


def Throughput(agent_reward, tdma_reward):
    max_iter = 100
    N = 20

    throughput_agent = np.zeros((1, max_iter))
    throughput_tdma = np.zeros((1, max_iter))
    total_optimal = np.ones(max_iter) * 1

    agent_temp_sum = 0
    tdma_temp_sum = 0
    for i in range(0, max_iter):
        if i < N:
            agent_temp_sum += agent_reward[i]
            tdma_temp_sum += tdma_reward[i]
            throughput_agent[0][i] = agent_temp_sum / (i + 1)
            throughput_tdma[0][i] = tdma_temp_sum / (i + 1)
        else:
            agent_temp_sum += agent_reward[i] - agent_reward[i - N]
            tdma_temp_sum += tdma_reward[i] - tdma_reward[i - N]
            throughput_agent[0][i] = agent_temp_sum / N
            throughput_tdma[0][i] = tdma_temp_sum / N
    return throughput_tdma.squeeze()
