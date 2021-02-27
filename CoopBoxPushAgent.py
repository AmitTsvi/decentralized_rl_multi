from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import pyspiel
import numpy as np
from absl import app
import torch

game_name = "coop_box_pushing"
players = None
load_state = None

def state_to_board_tensor(state):
    s = state.__str__()
    lines_list = s.splitlines()[3:]



def main(_):
    action_string = None

    print("Creating game: " + game_name)
    if players is not None:
        game = pyspiel.load_game(game_name, {"players": pyspiel.GameParameter(players)})
    else:
        game = pyspiel.load_game("coop_box_pushing", {"fully_observable":pyspiel.GameParameter(True), "horizon":pyspiel.GameParameter(100)})

    # Get a new state
    if load_state is not None:
        # Load a specific state
        state_string = ""
        with open(load_state, encoding="utf-8") as input_file:
            for line in input_file:
                state_string += line
        state_string = state_string.rstrip()
        print("Loading state:")
        print(state_string)
        print("")
        state = game.deserialize_state(state_string)
    else:
        state = game.new_initial_state()

    # Print the initial state
    print(str(state))

    # Get state board as tensor
    t = state.observation_tensor(0)
    t = torch.tensor(t)
    t.reshape(11, 8, 8)


    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            print("Sampled outcome: ",
                  state.action_to_string(state.current_player(), action))
            state.apply_action(action)

        elif state.is_simultaneous_node():
            # Simultaneous node: sample actions for all players.
            chosen_actions = [
                random.choice(state.legal_actions(pid))
                for pid in range(game.num_players())
            ]
            print("Chosen actions: ", [
                state.action_to_string(pid, action)
                for pid, action in enumerate(chosen_actions)
            ])
            state.apply_actions(chosen_actions)

        else:
            # Decision node: sample action for the single current player
            action = random.choice(state.legal_actions(state.current_player()))
            action_string = state.action_to_string(state.current_player(), action)
            print("Player ", state.current_player(), ", randomly sampled action: ",
                  action_string)
            state.apply_action(action)

        print(str(state))

    # Game is now done. Print utilities for each player
    returns = state.returns()
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid, returns[pid]))

# game.max_game_length() = 100
# game.num_distinct_actions() = 4 (Left, Right, Forward, Stay)
# game.num_players() = 2 (players)
# game.observation_tensor_shape() = [5] (probably the 5 values a board tile can have)
# state = game.new_initial_state()
#
# state.observation_tensor(id) - one-hot vector observation of player #id
# state.apply_action()
# state.current_player() = -1 or -2
# state.get_game() returns the game object
# state.is_terminal()
# state.is_chance_node()
# state.legal_actions() - list of legal actions
# state.rewards() - list of players' rewards from current turn
# state.returns() - list of player's sum of rewards up to this point
# state.num_distinct_actions() = 4 (forward, left, right, stay)
# state.num_players() = 2 (players)
# state.history_str() - all actions performed

# If we want deterministic game then the following should be played:
# >>> state.apply_actions([player1_action,player2_action])
# >>> state.apply_action(0) Success for player1 action
# >>> state.apply_action(0) Success for player2 action
# >>> state.apply_action(2) A constant order of actions applied (e.g. when both try to move to the same spot)

# env needs to implement the following:
# state = env.reset()
# self.state_dim = self.env.observation_space.shape[0]
# self.action_dim = self.env.action_space.n if self.is_disc_action else self.env.action_space.shape[0]
# self.max_episode_length = self.env._max_episode_steps

if __name__ == "__main__":
    app.run(main)
