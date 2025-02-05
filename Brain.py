import numpy as np
import os
# GPU Run
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.compat.v1.Session(config=config))

from keras.models import Model
from keras.layers import Dense, Input, Add
from keras.optimizers import RMSprop
from keras.initializers import glorot_normal

# DQN algorithm
class DQN:
    def __init__(self, state_size, n_actions, n_nodes,
                 state_action_memory_size, memory_size=500, replace_target_iter=200, batch_size=32, learning_rate=0.01, # 学习率
                 gamma=0.9, epsilon=1, epsilon_min=0.01, epsilon_decay=0.995):
        # hyper-parameters
        self.state_size = state_size
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.state_action_memory_size = state_action_memory_size
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_action_memory = np.zeros((self.state_action_memory_size, self.state_size + 1))
        self.memory = np.zeros((self.memory_size, self.state_size * 2 + 2))  # memory_size * len(s, a, r, s_)
        # temporary parameters
        self.learn_state_action_counter = 0
        self.learn_step_counter = 0
        self.memory_couter = 0

        # # # # # # # build mode
        self.model = self.build_DenseNet_model()  # model: evaluate Q value
        self.target_model = self.build_DenseNet_model()  # target_mode: target network

    # neural network
    def build_DenseNet_model(self):
        inputs = Input(shape=(self.state_size,))
        h1 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=247))(inputs)  # h1
        h2 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=2407))(h1)  # h2

        h3 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=2403))(h2)  # h3
        h4 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=24457))(h3)  # h4
        add1 = Add()([h4, h2])
        h5 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=24657))(add1)  # h5
        h6 = Dense(64, activation="relu", kernel_initializer=glorot_normal(seed=27567))(h5)  # h6
        add2 = Add()([h6, add1])

        outputs = Dense(self.n_actions, kernel_initializer=glorot_normal(seed=242147))(add2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
        return model

    # choose an action
    def choose_action(self, state):
        state = state[np.newaxis, :]
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        action_values = self.model.predict(state)
        return np.argmax(action_values)

    # experience buffer
    def store_transition(self, s, a, r, s_):  # s_: next_state
        if not hasattr(self, 'memory_couter'):
            self.memory_couter = 0
        transition = np.concatenate((s, [a, r], s_))
        index = self.memory_couter % self.memory_size
        self.memory[index, :] = transition
        self.memory_couter += 1

    # update Q-target
    def repalce_target_parameters(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    # train
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.repalce_target_parameters()  # iterative target model
        self.learn_step_counter += 1

        # sample batch memory from all memory
        if self.memory_couter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_couter, size=self.batch_size)

        # judicious experience replay
        sample_index1 = []
        sample_index2 = []
        for i in sample_index:
            if i >= self.memory_size-(self.state_action_memory_size):
                if self.memory_couter > self.memory_size:
                    a11 = np.random.randint(0, self.memory_size-(self.state_action_memory_size))
                    sample_index1.append(a11)
                    sample_index2.append(a11 + self.state_action_memory_size - 1)
                else:
                    a12 = np.random.randint(0, self.memory_couter-(self.state_action_memory_size))
                    sample_index1.append(a12)
                    sample_index2.append(a12 + self.state_action_memory_size - 1)
            else:
                sample_index1.append(i)
                sample_index2.append(i + self.state_action_memory_size - 1)

        batch_memory1 = self.memory[sample_index1, :]
        batch_memory2 = self.memory[sample_index2, :]

        # batch memory row: [s, a, r1, r2, s_]
        # number of batch memory: batch size
        # extract state, action, reward, reward2, next_state from bathc memory
        state = batch_memory1[:, :self.state_size]
        action = batch_memory1[:, self.state_size].astype(int)
        reward = batch_memory2[:, self.state_size + 1]
        next_state = batch_memory2[:, -self.state_size:]

        q_eval = self.model.predict(state)
        q_next = self.target_model.predict(next_state)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action] = reward + self.gamma * np.max(q_next, axis=1)
        self.model.fit(state, q_target, self.batch_size, epochs=1, verbose=0)