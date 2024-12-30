getAction(self, state, check_validity = True):
        observation = torch.tensor(processObs(state), dtype = torch.float32).to(self.q.device)
        q = self.q.forward(observation)
        action = int(torch.argmax(q))
        error = 0

        if check_validity:
            # checks for action validity
            valid_actions = state.get_valid_moves
            valid_actions_int = list(map(self.convert_to_int, valid_actions))

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