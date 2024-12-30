from state import State , State_2 , UltimateTTT_Move

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import Linear, ReLU, Dropout, BatchNorm1d
import os
import tqdm
import random_agent

## agent

def processObs(observation):
    valid_large_cell_move = np.zeros(9)  # Tạo mảng 9 phần tử, mỗi phần tử là 0
    for i in range(9):
        # Kiểm tra xem khối lớn có thể có nước đi hợp lệ không (kiểm tra bất kỳ ô trống nào trong khối nhỏ)
        if np.any(observation.blocks[i] == 0):  
            valid_large_cell_move[i] = 1  # Nếu có ô trống trong khối nhỏ, gán giá trị 1

    # Flatten tất cả các blocks thành 1D mảng và kết hợp với global_cells và valid_large_cell_move
    blocks_flattened = np.concatenate([block.flatten() for block in observation.blocks])
    
    # Kết hợp global_cells, các blocks đã flatten và valid_large_cell_move thành 1 mảng duy nhất
    return np.concatenate([observation.global_cells, blocks_flattened, valid_large_cell_move]).flatten()




class ReplayBuffer(object):
    def __init__(self, state_len, mem_size):
        self.state_len = state_len
        self.mem_size = mem_size
        self.mem_counter = 0
        self.states = np.zeros((mem_size, state_len), dtype=np.float32)
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.new_states = np.zeros((mem_size, state_len), dtype=np.float32)
        self.dones = np.zeros(mem_size, dtype=np.int32)


    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter%self.mem_size
        self.states[index, :] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index, :] = new_state
        self.dones[index] = done
        self.mem_counter += 1

    def sample_memory(self, batch_size):
        max_memory = min(self.mem_size, self.mem_counter)
        batch = np.random.choice(np.arange(max_memory), batch_size, replace=False)
        states = self.states[batch, :]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch, :]
        dones = self.dones[batch]
        return states, actions, rewards, new_states, dones


class DQNetwork(torch.nn.Module):
    def __init__(self, state_len, n_actions,learning_rate):
        super(DQNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #print(self.device)
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_len, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )
        self.optimizer = Adam(self.parameters(), lr = learning_rate)
        self.loss = torch.nn.MSELoss(reduction='sum')
        self.to(self.device)


    def forward(self,state):
        return self.network(state)

    def save(self, name = ""):
        torch.save(self,os.path.dirname(__file__)+"/DQNetwork" + name + ".pt")

    def load(self, name = ""):
        self = torch.load(os.path.dirname(__file__)+"/DQNetwork" + name + ".pt")
        self.eval()


class DQNAgent():

    def __init__(self, loading=True, name = ""):

        learning_rate=0.00005
        gamma=0.99
        batch_size=64
        state_len=99
        mem_size = 1000000
        min_memory_for_training=1000
        epsilon=0.5
        epsilon_dec=0.998
        epsilon_min = 0.1
        frozen_iterations=6


        self.it_counter = 0            # how many timesteps have passed already
        self.gamma = gamma             # gamma hyperparameter
        self.batch_size = batch_size   # batch size hyperparameter for neural network
        self.state_len = state_len     # how long the state vector is
        self.epsilon = epsilon         # epsilon start value (1=completly random)
        self.epsilon_min = epsilon_min # the minimum value
        self.epsilon_dec = epsilon_dec
        self.mem_size = mem_size

        self.min_memory_for_training = min_memory_for_training
        self.q = DQNetwork(state_len, 81, learning_rate)
        self.replay_buffer = ReplayBuffer(self.state_len, mem_size)

        if loading :
            self.q.load(name)

    def convert_to_int(self, action : UltimateTTT_Move):
        return action.index_local_board*9 + action.x*3 + action.y
    
    def convert_to_ultimatettt_move(self, value : int, state : State):

        index_local_board = value // 9  # Xác định bảng nhỏ (0-8)
        local_index = value % 9        # Chỉ số trong bảng nhỏ (0-8)
        x_coordinate = local_index // 3 # Tọa độ x (0-2)
        y_coordinate = local_index % 3 # Tọa độ y (0-2)

        return UltimateTTT_Move(index_local_board, x_coordinate, y_coordinate, state.player_to_move)
    
    def pickActionMaybeMasked(self, state):
        if np.random.random() < self.epsilon:
            return self.getAction(state, True)
        else:
            return self.getAction(state, False)
        
    def getAction(self, state, check_validity = True):
        observation = torch.tensor(processObs(state), dtype = torch.float32).to(self.q.device)
        q = self.q.forward(observation)
        action = int(torch.argmax(q))
        error = 0
        valid_actions = state.get_valid_moves
        valid_actions_int = list(map(self.convert_to_int, valid_actions))
        if check_validity:
            # checks for action validity
            if action in valid_actions_int:
                pass
            else:
                q_min = float(torch.min(q))
                mask = np.array([True if i in valid_actions_int else False for i in range(81)])
                new_q = (q.detach().cpu().numpy() - q_min + 1.) *  mask
                action = int(np.argmax(new_q))
                
        if action not in valid_actions_int: 
            error = -100
        # value = q[action]
        return error, self.convert_to_ultimatettt_move(action, state)
        

    def learn(self, error):
        if self.replay_buffer.mem_counter < self.min_memory_for_training:
            return
        states, actions, rewards, new_states, dones = self.replay_buffer.sample_memory(self.batch_size)
        self.q.optimizer.zero_grad()
        states_batch = torch.tensor(states, dtype = torch.float32).to(self.q.device)
        new_states_batch = torch.tensor(new_states,dtype = torch.float32).to(self.q.device)
        actions_batch = torch.tensor(actions, dtype = torch.long).to(self.q.device)
        rewards_batch = torch.tensor(rewards, dtype = torch.float32).to(self.q.device)
        dones_batch = torch.tensor(dones, dtype = torch.float32).to(self.q.device)

        target = rewards_batch + torch.mul(self.gamma* self.q(new_states_batch).max(axis = 1).values, (1 - dones_batch))
        prediction = self.q.forward(states_batch).gather(1,actions_batch.unsqueeze(1)).squeeze(1)

        loss = self.q.loss(prediction, target)
        loss.backward()  # Compute gradients
        self.q.optimizer.step()  # Backpropagate error

        # decrease epsilon:
        if error == 0:
            if self.epsilon * self.epsilon_dec > self.epsilon_min:
                self.epsilon *= self.epsilon_dec
        else:
            if self.epsilon / self.epsilon_dec <= 1:
                self.epsilon /= self.epsilon_dec

        self.it_counter += 1
        return

    def learnNN(self, n_episodes=100, trainingName="", play_first=True):
        l_epsilon = []
        sum_win = 0

        for episode in tqdm.tqdm(range(n_episodes)):
            state = State()  # Resetting the state after each episode
            is_game_done = False
            
            while not state.game_over and not is_game_done:
                error = 0
                new_move = None
                # Step 1: AgentDQN's turn
                if state.player_to_move == 1:
                    if play_first:
                        error, new_move = self.getAction(state)
                    else:
                        new_move = random_agent.select_move(state, 100)
                else:
                    if play_first:
                        new_move = random_agent.select_move(state, 100)
                    else:
                        error, new_move = self.getAction(state)
                                    
                if not new_move or error != 0:
                    is_game_done = True
                    continue
                
                # Execute move
                prev_state = state
                state.act_move(new_move)


                # Step 2: Check if game is over before the next player's turn
                if state.game_over:
                    reward = error if error != 0 else -complex_utility(state)
                    self.replay_buffer.store_transition(
                        processObs(prev_state), 
                        self.convert_to_int(new_move), 
                        reward, 
                        processObs(state), 
                        state.game_over
                    )
                    self.learn(error)
                    break
                
                new_move2 = None
                # Step 3: Random agent's turn
                if state.player_to_move == 1:
                    if play_first:
                        error, new_move2 = self.getAction(state)
                    else:
                        new_move2 = random_agent.select_move(state, 100)
                else:
                    if play_first:
                        new_move2 = random_agent.select_move(state, 100)
                    else:
                        error, new_move2 = self.getAction(state)
                    
                if not new_move2 or error != 0:
                    is_game_done = True
                    continue
                    
                    # Execute move
                state.act_move(new_move2)
                    # Store experience and learn only for agentDQN
                self.replay_buffer.store_transition(
                    processObs(prev_state), 
                    self.convert_to_int(new_move), 
                    complex_utility(state), 
                    processObs(state), 
                    state.game_over
                )
                self.learn(error)

                # Check if the game is over after random agent's move
                if state.game_over:
                    break
                
            # Update results after the episode ends
            if state.game_result(state.global_cells.reshape(3, 3)) == 1:
                sum_win += 1
            elif state.game_result(state.global_cells.reshape(3, 3)) == -1:
                sum_win -= 1

            # Log epsilon and win metrics
            l_epsilon.append(self.epsilon)

        # Save the trained model
        self.q.save(trainingName)
        print(l_epsilon)
        print("\n")
        print("sum_win: ", sum_win)
        
    


def get_complex_value(state):
    value = 0

    lines1 = set()
    lines2 = set()
    used = set()

    # Lignes horizontales
    for i in range(3):
        nb1 = 0
        nb2 = 0
        zero = None
        addToUsed = False
        for j in range(3):
            index = 3*i+j
            if state[i][j] == 1: nb1 += 1
            elif state[i][j] == -1: nb2 += 1
            else: zero = index
        if nb1 == 2 and nb2 == 0:
            lines1.add(zero)
            addToUsed = True
        elif nb2 == 2 and nb1 == 0:
            lines2.add(zero)
            addToUsed = True
        if addToUsed:
            for j in range(3):
                used.add(3*i +j)

    # Lignes verticales
    for j in range(3):
        nb1 = 0
        nb2 = 0
        zero = None
        addToUsed = False
        for i in range(3):
            index = 3*i +j
            if state[i][j] == 1: nb1 += 1
            elif state[i][j] == -1: nb2 += 1
            else: zero = index
        if nb1 == 2 and nb2 == 0:
            lines1.add(zero)
            addToUsed = True
        elif nb2 == 2 and nb1 == 0:
            lines2.add(zero)
            addToUsed = True
        if addToUsed:
            for i in range(3):
                used.add(3*i +j)

            # Diagonale 1
    nb1 = 0
    nb2 = 0
    zero = None
    addToUsed = False
    for i in range(3):
        index = 3*i +i
        if state[i][i] == 1: nb1 += 1
        elif state[i][i] == -1: nb2 += 1
        else: zero = index
    if nb1 == 2 and nb2 == 0:
        lines1.add(zero)
        addToUsed = True
    elif nb2 == 2 and nb1 == 0:
        lines2.add(zero)
        addToUsed = True
    if addToUsed:
        for i in range(3):
            used.add(3*i +i)

            # Diagonale 2
    nb1 = 0
    nb2 = 0
    zero = None
    addToUsed = False
    for i in range(3):
        index = 3*(2-i) +i
        if state[i][2-i] == 1: nb1 += 1
        elif state[i][2-i] == -1: nb2 += 1
        else: zero = index
    if nb1 == 2 and nb2 == 0:
        lines1.add(zero)
        addToUsed = True
    elif nb2 == 2 and nb1 == 0:
        lines2.add(zero)
        addToUsed = True
    if addToUsed:
        for i in range(3):
            used.add(3*(2-i) +i)

    value += 5 * min(2, len(lines1))
    value -= 5 * min(2, len(lines2))

    for i in range(3):
        for j in range(3):
            index =  3*i + j
            if state[i][j] != 0 and not(index in used):
                val = 1 # edge
                if i == j == 1: # Middle
                    val = 2
                elif (i != 1) and (j != 1): # Corner
                    val = 1.5
                value += val * (state[i][j])
    return value

def complex_utility(state: State):
    result = 0
    if state.game_result(state.global_cells.reshape(3,3)):
        result = 400*state.game_result(state.global_cells.reshape(3,3))
    

    large_board_index = 3*state.previous_move.x + state.previous_move.y
    u_small_cur = get_complex_value(state.blocks[large_board_index])
    u_large_cur = get_complex_value(state.global_cells.reshape(3,3))

    if state.game_result(state.blocks[large_board_index]):  
        result = 20 * (u_large_cur)
    else:
        result =  (u_large_cur)*abs(u_small_cur)
    if state.player_to_move == 1: return result
    else: return -result

def train_agent_with_learnNN(n_episodes, trainingName):
    # Tạo agent
    agent = DQNAgent(loading=True, name="_final")

    # Gọi hàm learnNN
    agent.learnNN(
        n_episodes=n_episodes,
        trainingName=trainingName,
        play_first=False
    )

train_agent_with_learnNN(n_episodes=12000, trainingName="_final")


def select_move(cur_state, remain_time):
    DQNagent = DQNAgent(loading=True, name="_final")
    error, move = DQNagent.getAction(cur_state)
    if error == 0: return move
    return None

    