import numpy as np
from state import State , UltimateTTT_Move
import math
import copy
import random

def complex_utility(state: State):
    if state.game_result(state.global_cells.reshape(3,3)):
        return 400*state.game_result(state.global_cells.reshape(3,3))
    
    large_board_index = 3*state.previous_move.x + state.previous_move.y

    u_small_cur = get_complex_value(state.blocks[large_board_index])
    #print('u_small_cur', u_small_cur)
    u_large_cur = get_complex_value(state.global_cells.reshape(3,3))
    #print('u_large_cur', u_large_cur)

    if state.game_result(state.blocks[large_board_index]):
        return 20 * (u_large_cur)
    else:
        return (u_large_cur)*abs(u_small_cur)

        
def simple_utility(state : State):
    if state.game_result(state.global_cells.reshape(3,3)):
        return 400*state.game_result(state.global_cells.reshape(3,3))
    else:
        large_board_index = 3*state.previous_move.x + state.previous_move.y 
        if state.game_result(state.blocks[large_board_index]):
            return 100*state.game_result(state.blocks[large_board_index])
    return 0
        



isTheFirstTurn = True

def select_move(cur_state : State, remain_time):
    global isTheFirstTurn
    if isTheFirstTurn and cur_state.player_to_move == 1:
        isTheFirstTurn = False
        return UltimateTTT_Move(4,1,1,1)
    else:
        num_steps = 0
        for i in range(9):
            for j in range(3):
                for k in range(3):
                    if cur_state.blocks[i][j][k] != 0:
                        num_steps += 1
        valid_moves = cur_state.get_valid_moves
        if len(valid_moves) != 0:
            player = cur_state.player_to_move == 1
            if num_steps >= 25:
                value , move = alpha_beta(cur_state,5,-math.inf,math.inf,player,2)
                #print('best value ', value)
                return move
            else:
                value , move = alpha_beta(cur_state,4,-math.inf,math.inf,player,2)
                #print('best value ', value)
                return move
    return None

def alpha_beta(cur_state: State, depth, alpha, beta, isMaxPlayer, utility_mode):
    if depth == 0 or cur_state.game_over:
        if utility_mode == 1:
            return simple_utility(cur_state), None
            #print('Using simple utility')
        else:
            return complex_utility(cur_state), None
            #print('Using complex utility')
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) == 0: return 9999, None

    newDepth = depth - 1
    if len(valid_moves) > 9: newDepth = max(0, depth - 2)

    if isMaxPlayer:
        best_value = -math.inf
        max_move = None
        for move in valid_moves:
            new_state = copy.deepcopy(cur_state)
            new_state.act_move(move)
            value, _ = alpha_beta(new_state, newDepth, alpha, beta, False, utility_mode)
            if value > best_value:
                best_value = value
                max_move = [move]
            elif value == best_value:
                max_move.append(move)
            if value > beta:
                break
            alpha = max(value, alpha)
        if max_move is None:
            return best_value, None
        return best_value, random.choice(max_move)
    else:
        best_value = math.inf
        min_move = None
        for move in valid_moves:
            new_state = copy.deepcopy(cur_state)
            new_state.act_move(move)
            value, _ = alpha_beta(new_state, newDepth, alpha, beta, True, utility_mode)
            if value < best_value:
                best_value = value
                min_move = [move]
            elif value == best_value:
                min_move.append(move)
            if value < alpha:
                break
            beta = min(value, beta)
        if min_move is None:
            return best_value, None
        return best_value, random.choice(min_move)

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
    
    