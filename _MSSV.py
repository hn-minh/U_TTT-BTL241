import numpy as np
from state import State , UltimateTTT_Move
import math
import copy

isTheFirstTurn = True

def select_move(cur_state : State, remain_time):
    global isTheFirstTurn
    if isTheFirstTurn and cur_state.player_to_move == 1:
        isTheFirstTurn = False
        return UltimateTTT_Move(4,1,1,1)
    else:
        valid_moves = cur_state.get_valid_moves
        if len(valid_moves) != 0:
            player = cur_state.player_to_move == 1
            _ , move = alpha_beta(cur_state,6,-math.inf,math.inf,player)
            return move
    return None

def alpha_beta(cur_state: State, depth, alpha, beta, isMaxPlayer):
    if depth == 0 or cur_state.game_over:
        return utility(cur_state), None 
    best_move = None
    if isMaxPlayer:
        best_value = -math.inf
        valid_moves = cur_state.get_valid_moves
        for move in valid_moves:
            new_state = copy.deepcopy(cur_state)
            new_state.act_move(move)
            value, _ = alpha_beta(State(state=new_state), depth - 1, alpha, beta, False)
            if value > best_value:
                best_value = value
                best_move = move
            if value > beta:
                break
            alpha = max(value, alpha)
        return best_value, best_move
    else:
        best_value = math.inf
        valid_moves = cur_state.get_valid_moves
        for move in valid_moves:
            new_state = copy.deepcopy(cur_state)
            new_state.act_move(move)
            value, _ = alpha_beta(State(state=new_state), depth - 1, alpha, beta, True)
            if value < best_value:
                best_value = value
                best_move = move
            if value < alpha:
                break
            beta = min(value, beta)
        return best_value, best_move
    
def utility(state : State):
    if state.game_result(state.global_cells.reshape(3,3)) == state.player_to_move:
        return 100 * (-1 * state.player_to_move)
    local_board = state.blocks[state.previous_move.index_local_board]
    if state.game_result(local_board):
        if state.previous_move.index_local_board == 4:
             return 50 * (-1 * state.player_to_move)
        else:
            return 10 * (-1 * state.player_to_move)
    if state.previous_move.x == 1 and state.previous_move.y == 1:
        return 10 * state.player_to_move
    return np.random.normal(loc=0,scale=10) * (-1 * state.player_to_move)
    