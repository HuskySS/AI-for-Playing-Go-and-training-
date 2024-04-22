import numpy as np
import paddle
class PolicyValueNet(paddle.nn.Layer):
    def __init__(self, input_channels: int = 10,
                 board_size: int = 9):

        super(PolicyValueNet, self).__init__()

        self.conv_layer = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            paddle.nn.ReLU()
        )

        self.policy_layer = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=128, out_channels=8, kernel_size=1),
            paddle.nn.ReLU(),
            paddle.nn.Flatten(),
            paddle.nn.Linear(in_features=9*9*8, out_features=256),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=256, out_features=board_size*board_size+1),
            paddle.nn.Softmax()
        )

        self.value_layer = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=128, out_channels=4, kernel_size=1),
            paddle.nn.ReLU(),
            paddle.nn.Flatten(),
            paddle.nn.Linear(in_features=9*9*4, out_features=128),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=128, out_features=64),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=64, out_features=1),
            paddle.nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        policy = self.policy_layer(x)
        value = self.value_layer(x)
        return policy, value

    def policy_value_fn(self, simulate_game_state):

        legal_positions = simulate_game_state.valid_move_idcs()
        current_state = paddle.to_tensor(simulate_game_state.get_board_state()[np.newaxis], dtype='float32')
        act_probs, value = self.forward(current_state)
        act_probs = zip(legal_positions, act_probs.numpy().flatten()[legal_positions])
        return act_probs, value
