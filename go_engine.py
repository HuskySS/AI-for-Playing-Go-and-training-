from GoImplementation.goImple import govars, gogame
from typing import Union, List, Tuple
import numpy as np
from scipy import ndimage

surround_struct = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])

eye_struct = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

corner_struct = np.array([[1, 0, 1],
                          [0, 0, 0],
                          [1, 0, 1]])
BLACK = govars.BLACK
WHITE = govars.WHITE


class GoEngine:
    def __init__(self, board_size: int = 9,
                 komi=7.5,
                 record_step: int = 4,
                 state_format: str = "separated",
                 record_last: bool = True):

        assert state_format in ["separated", "merged"],\
            "state_format can only be 'separated' or 'merged', but received: {}".format(state_format)

        self.board_size = board_size
        self.komi = komi
        self.record_step = record_step
        self.state_format = state_format
        self.record_last = record_last
        self.current_state = gogame.init_state(board_size)
        self.board_state_history = []
        self.action_history = []

        if state_format == "separated":
            record_step *= 2
        self.state_channels = record_step + 2 if record_last else record_step + 1
        self.board_state = np.zeros((self.state_channels, board_size, board_size))
        self.done = False

    def reset(self) -> np.ndarray:
        self.current_state = gogame.init_state(self.board_size)
        self.board_state = np.zeros((self.state_channels, self.board_size, self.board_size))
        self.board_state_history = []
        self.action_history = []
        self.done = False
        return np.copy(self.current_state)

    def step(self, action: Union[List[int], Tuple[int], int, None]) -> np.ndarray:

        assert not self.done
        if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
            assert 0 <= action[0] < self.board_size
            assert 0 <= action[1] < self.board_size
            action = self.board_size * action[0] + action[1]
        elif isinstance(action, int):
            assert 0 <= action <= self.board_size ** 2
        elif action is None:
            action = self.board_size ** 2

        self.current_state = gogame.next_state(self.current_state, action, canonical=False)
        self.board_state = self._update_state_step(action)
        self.board_state_history.append(np.copy(self.current_state))
        self.action_history.append(action)
        self.done = gogame.game_ended(self.current_state)
        return np.copy(self.current_state)

    def _update_state_step(self, action: int) -> np.ndarray:
        if self.state_format == "separated":
            if self.turn() == govars.WHITE:
                self.board_state[:self.record_step - 1] = np.copy(self.board_state[1:self.record_step])
                self.board_state[self.record_step - 1] = np.copy(self.current_state[govars.BLACK])
            else:
                self.board_state[self.record_step: self.record_step * 2 - 1] = \
                    np.copy(self.board_state[self.record_step + 1: self.record_step * 2])
                self.board_state[self.record_step * 2 - 1] = np.copy(self.current_state[govars.WHITE])
        elif self.state_format == "merged":
            self.board_state[:self.record_step - 1] = np.copy(self.board_state[1:self.record_step])
            current_state = self.current_state[[govars.BLACK, govars.WHITE]]
            current_state[govars.WHITE] *= -1
            self.board_state[self.record_step - 1] = np.sum(current_state, axis=0)

        if self.record_last:
            self.board_state[-2] = np.copy(self.current_state[govars.TURN_CHNL])
            self.board_state[-1] = np.zeros((self.board_size, self.board_size))
            if action != self.board_size ** 2:
                position = action // self.board_size, action % self.board_size
                self.board_state[-1, position[0], position[1]] = 1
        else:
            self.board_state[-1] = np.copy(self.current_state[govars.TURN_CHNL])
        return np.copy(self.board_state)

    def get_board_state(self) -> np.ndarray:
        return np.copy(self.board_state)

    def game_ended(self) -> bool:
        return self.done

    def winner(self) -> int:
        if not self.done:
            return -1
        else:
            winner = self.winning()
            winner = govars.BLACK if winner == 1 else govars.WHITE
            return winner

    def action_valid(self, action) -> bool:
        return self.valid_moves()[action]

    def valid_move_idcs(self) -> np.ndarray:
        valid_moves = self.valid_moves()
        return np.argwhere(valid_moves).flatten()

    def advanced_valid_move_idcs(self) -> np.ndarray:
        advanced_valid_moves = self.advanced_valid_moves()
        return np.argwhere(advanced_valid_moves).flatten()

    def uniform_random_action(self) -> np.ndarray:
        valid_move_idcs = self.valid_move_idcs()
        return np.random.choice(valid_move_idcs)

    def advanced_uniform_random_action(self) -> np.ndarray:
        advanced_valid_move_idcs = self.advanced_valid_move_idcs()
        return np.random.choice(advanced_valid_move_idcs)

    def turn(self) -> int:
        return gogame.turn(self.current_state)

    def valid_moves(self) -> np.ndarray:
        return gogame.valid_moves(self.current_state)

    def advanced_valid_moves(self):
        valid_moves = 1 - self.current_state[govars.INVD_CHNL]
        eyes_mask = 1 - self.eyes()
        return np.append((valid_moves * eyes_mask).flatten(), 1)

    def winning(self):

        return gogame.winning(self.current_state, self.komi)

    def areas(self):
        return gogame.areas(self.current_state)

    def eyes(self):

        board_shape = self.current_state.shape[1:]

        side_mask = np.zeros(board_shape)
        side_mask[[0, -1], :] = 1
        side_mask[:, [0, -1]] = 1
        nonside_mask = 1 - side_mask

        next_player = self.turn()
        next_player_pieces = self.current_state[next_player]
        all_pieces = np.sum(self.current_state[[govars.BLACK, govars.WHITE]], axis=0)
        empties = 1 - all_pieces

        side_matrix = ndimage.convolve(next_player_pieces, eye_struct, mode='constant', cval=1) == 8
        side_matrix = side_matrix * side_mask
        nonside_matrix = ndimage.convolve(next_player_pieces, surround_struct, mode='constant', cval=1) == 4
        nonside_matrix *= ndimage.convolve(next_player_pieces, corner_struct, mode='constant', cval=1) > 2
        nonside_matrix = nonside_matrix * nonside_mask

        return empties * (side_matrix + nonside_matrix)

    def all_symmetries(self) -> List[np.ndarray]:
        return gogame.all_symmetries(np.copy(self.board_state))

    @staticmethod
    def array_symmetries(array: np.ndarray) -> List[np.ndarray]:

        return gogame.all_symmetries(array)
