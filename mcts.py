import numpy as np
import copy
from operator import itemgetter


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def evaluate_rollout(simulate_game_state, rollout_policy_fn, limit=1000):
    game_state_copy = copy.deepcopy(simulate_game_state)
    player = game_state_copy.turn()
    for _ in range(limit):
        end, winner = game_state_copy.game_ended(), game_state_copy.winner()
        if end:
            break
        action_probs = rollout_policy_fn(game_state_copy)
        max_action = max(action_probs, key=itemgetter(1))[0]
        game_state_copy.step(max_action)
    else:
        winner = -1
    if winner == -1:
        return 0
    else:
        return 1 if winner == player else -1


class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.U = 0
        self.P = prior_p

    def select(self, c_puct):

        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors):

        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def update(self, leaf_value):

        self.n_visits += 1
        # Q =  W / N
        # Q_old = W_old / N_old
        # Q = (W_old + v) / (N_old + 1) = (Q_old * N_old + v) / (N_old + 1)
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):

        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):

        self.U = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.U

    def is_leaf(self):

        return self.children == {}

    def is_root(self):

        return self.parent is None


class MCTS:

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_fn

        self.c_puct = c_puct
        self.n_playout = n_playout

    def playout(self, simulate_game_state):

        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            simulate_game_state.step(action)

        action_probs, leaf_value = self.policy(simulate_game_state)

        end, winner = simulate_game_state.game_ended(), simulate_game_state.winner()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == simulate_game_state.turn() else -1.0
                )

        node.update_recursive(-leaf_value)

    def get_move_probs(self, game, temp=1e-3, player=None):

        for i in range(self.n_playout):
            if not player.valid:
                return -1, -1
            if player is not None:
                player.speed = (i + 1, self.n_playout)
            simulate_game_state = game.game_state_simulator(player.is_selfplay)
            self.playout(simulate_game_state)
        act_visits = [(act, node.n_visits)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def get_move(self, game, player=None):

        for i in range(self.n_playout):
            if not player.valid:
                return -1
            if player is not None:
                player.speed = (i + 1, self.n_playout)
            game_state = game.game_state_simulator()
            self.playout(game_state)
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]

    def update_with_move(self, last_move):

        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
