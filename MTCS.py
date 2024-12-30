import numpy as np
import math
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from state import State , State_2 , UltimateTTT_Move

# Define the neural network for policy and value prediction
class PolicyValueNet(nn.Module):
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 9 * 9, 81),  # 9x9 board flattened
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

# TreeNode and MonteCarloTreeSearch are modified to integrate with the neural network
class TreeNode:
    def __init__(self, cur_state, action, parent_node, prior_prob=0):
        self.cur_state = cur_state
        self.action = action
        self.parent_node = parent_node
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior_prob = prior_prob

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, valid_moves, policy_probs):
        for move, prob in zip(valid_moves, policy_probs):
            new_state = copy.deepcopy(self.cur_state)
            new_state.act_move(move)
            self.children.append(TreeNode(new_state, move, self, prob))

    def update(self, value):
        self.visits += 1
        self.value += (value - self.value) / self.visits

    def select_child(self, exploration_constant):
        best_value = -float('inf')
        best_child = None

        for child in self.children:
            uct_value = (
                child.value +
                exploration_constant * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)
            )
            if uct_value > best_value:
                best_value = uct_value
                best_child = child

        return best_child

class MonteCarloTreeSearch:
    def __init__(self, model, exploration_constant=1.4):
        self.model = model
        self.exploration_constant = exploration_constant

    def search(self, root, num_simulations):
        for _ in range(num_simulations):
            node = root

            # Selection
            while not node.is_leaf():
                node = node.select_child(self.exploration_constant)

            # Evaluation
            state_tensor = torch.tensor(node.cur_state.to_tensor(), dtype=torch.float32).unsqueeze(0)
            policy, value = self.model(state_tensor)
            policy = policy.squeeze().detach().numpy()
            value = value.item()

            # Expansion
            valid_moves = node.cur_state.get_valid_moves()
            if valid_moves:
                node.expand(valid_moves, policy[valid_moves])

            # Backpropagation
            while node is not None:
                node.update(value)
                node = node.parent_node

        # Choose the best move
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

# Replace `select_move` with the following function
isTheFirstTurn = True
model = PolicyValueNet()
model.eval()  # Ensure model is in evaluation mode

def select_move(cur_state, remain_time):
    global isTheFirstTurn
    if isTheFirstTurn and cur_state.player_to_move == 1:
        isTheFirstTurn = False
        return UltimateTTT_Move(4, 1, 1, 1)
    else:
        root = TreeNode(cur_state, None, None)
        mcts = MonteCarloTreeSearch(model)
        move = mcts.search(root, num_simulations=100)
        if move is not None and cur_state.is_valid_move(move):
            return move
        return None
