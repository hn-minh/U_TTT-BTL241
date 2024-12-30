import numpy as np

def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        random_move = np.random.choice(valid_moves)
        #list_int = []
        #for i in valid_moves:
            #list_int.append(i.index_local_board*9 + i.x*3 + i.y)
        #random_int = random_move.index_local_board*9 + random_move.x*3 + random_move.y
        return random_move
    return None
