from threading import Thread
import numpy as np
from time import sleep
from mcts import MCTS, evaluate_rollout
from policy_value_net import PolicyValueNet
import paddle
import os


class Player:
    def __init__(self):
        self.allow = True
        self.action = None
        self.name = 'Player'
        self.valid = True
        self.speed = None

    def play(self, game):
        if self.allow and self.action is None:
            self.allow = False
            Thread(target=self.step, args=(game, ), daemon=True).start()

    def step(self, game):

        print('Hello!')


class HumanPlayer(Player):
    def __init__(self):
        super().__init__()
        self.name = 'human'


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()
        self.name = 'random'

    def step(self, game):
        sleep(1)
        self.action = self.get_action(game)

    @staticmethod
    def get_action(game):
        valid_move_idcs = game.game_state.advanced_valid_move_idcs()
        if len(valid_move_idcs) > 1:
            valid_move_idcs = valid_move_idcs[:-1]
        action = np.random.choice(valid_move_idcs)
        return action


class MCTSPlayer(Player):
    def __init__(self, c_puct=5, n_playout=20):
        super().__init__()
        self.name = 'MCTS{}'.format(n_playout)

        def rollout_policy_fn(game_state_simulator):
            availables = game_state_simulator.valid_move_idcs()
            action_probs = np.random.rand(len(availables))
            return zip(availables, action_probs)

        def policy_value_fn(game_state_simulator):
            availables = game_state_simulator.valid_move_idcs()
            action_probs = np.ones(len(availables)) / len(availables)
            return zip(availables, action_probs), evaluate_rollout(game_state_simulator, rollout_policy_fn)

        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def step(self, game):
        action = self.get_action(game)
        if action == -1:
            action = None
            self.allow = True
        self.action = action

        self.speed = (0, 1)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, game):
        move = self.mcts.get_move(game, self)
        self.mcts.update_with_move(-1)
        return move


class AlphaGoPlayer(Player):
    def __init__(self, model_path='models/pdparams', c_puct=5, n_playout=400, is_selfplay=False):
        super(AlphaGoPlayer, self).__init__()
        if model_path == 'models/alpha_go.pdparams':
            self.name = 'alfaGo'
        elif model_path == 'models/my_alpha_go.pdparams':
            self.name = 'myAlfaGo'
        else:
            self.name = 'Unexpected name'
        self.policy_value_net = PolicyValueNet()
        self.policy_value_net.eval()

        if os.path.exists(model_path):
            state_dict = paddle.load(model_path)
            self.policy_value_net.set_state_dict(state_dict)

        self.mcts = MCTS(self.policy_value_net.policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def step(self, game):
        action = self.get_action(game)
        if action == -1:
            action = None
            self.allow = True
        self.action = action
        self.speed = (0, 1)

    def get_action(self, game, temp=1e-3, return_probs=False):
        move_probs = np.zeros(game.board_size ** 2 + 1)
        acts, probs = self.mcts.get_move_probs(game, temp, self)
        if acts == -1 and probs == -1:
            return -1
        move_probs[list(acts)] = probs
        if self.is_selfplay:

            move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
        if return_probs:
            return move, move_probs
        else:
            return move


class PolicyNetPlayer(Player):
    def __init__(self, model_path='models/model.pdparams'):
        super(PolicyNetPlayer, self).__init__()
        self.name = 'policyNet'
        self.policy_value_net = PolicyValueNet()

        if os.path.exists(model_path):
            state_dict = paddle.load(model_path)
            self.policy_value_net.set_state_dict(state_dict)
        self.policy_value_net.eval()

    def step(self, game):
        sleep(1)
        self.action = self.get_action(game)

    def get_action(self, game):
        valid_moves = game.game_state.valid_moves()
        valid_moves = paddle.to_tensor(valid_moves)

        current_state = game.game_state.get_board_state()
        current_state = paddle.to_tensor([current_state], dtype='float32')
        probs, _ = self.policy_value_net(current_state)
        probs = probs[0]
        probs *= valid_moves
        probs = probs / paddle.sum(probs)

        action = np.random.choice(range(82), p=probs.numpy())
        return action


class ValueNetPlayer(Player):
    def __init__(self, model_path='models/model.pdparams'):
        super(ValueNetPlayer, self).__init__()
        self.name = 'valueNet'
        self.policy_value_net = PolicyValueNet()

        if os.path.exists(model_path):
            state_dict = paddle.load(model_path)
            self.policy_value_net.set_state_dict(state_dict)
        self.policy_value_net.eval()

    def step(self, game):
        sleep(1)
        self.action = self.get_action(game)

    def get_action(self, game):
        valid_move_idcs = game.game_state.valid_move_idcs()

        max_value = 1
        action = game.board_size ** 2
        for simulate_action in valid_move_idcs:
            simulate_game_state = game.game_state_simulator()
            simulate_game_state.step(simulate_action)

            current_state = simulate_game_state.get_board_state()
            current_state = paddle.to_tensor([current_state], dtype='float32')

            _, value = self.policy_value_net(current_state)
            value = value.numpy().flatten()[0]

            if value < max_value:
                max_value = value
                action = simulate_action

        return action
