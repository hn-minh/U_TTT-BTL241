from state import State, State_2
import time
from importlib import import_module
from DQN import DQNetwork

  
def main(player_X, player_O, rule = 1):
    dict_player = {1: 'X', -1: 'O'}
    if rule == 1:
        cur_state = State()
    else:
        cur_state = State_2()
    turn = 1    

    limit = 81
    remain_time_X = 120
    remain_time_O = 120
    
    player_1 = import_module(player_X)
    player_2 = import_module(player_O)
    
    
    while turn <= limit:
        #print("turn:", turn, end='\n\n')
        if cur_state.game_over:
            print("winner:", dict_player[cur_state.player_to_move * -1])
            break
        
        start_time = time.time()
        if cur_state.player_to_move == 1:
            new_move = player_1.select_move(cur_state, remain_time_X)
            elapsed_time = time.time() - start_time
            remain_time_X -= elapsed_time
        else:
            new_move = player_2.select_move(cur_state, remain_time_O)
            elapsed_time = time.time() - start_time
            remain_time_O -= elapsed_time
            
        if new_move == None:
            break
        
        if remain_time_X < 0 or remain_time_O < 0:
            print("out of time")
            print("winner:", dict_player[cur_state.player_to_move * -1])
            break
                
        if elapsed_time > 10.0:
            print("elapsed time:", elapsed_time)
            print("winner: ", dict_player[cur_state.player_to_move * -1])
            break
        
        cur_state.act_move(new_move)
        #print(cur_state)
        
        turn += 1
    if new_move == None: return 0
    if (cur_state.game_result(cur_state.global_cells.reshape(3,3)) == 1): return 1
    elif (cur_state.game_result(cur_state.global_cells.reshape(3,3)) == -1): return -1
    else: return 0
    #print("X:", cur_state.count_X)
    #print("O:", cur_state.count_O)
num_x_win = 0
num_o_win = 0
for i in range (1000):
    result = main('DQN', 'DQN')
    if result == 1: num_x_win += 1
    elif result == -1: num_o_win += 1
print("num x win ", num_x_win)
print("num o win ", num_o_win)

 
