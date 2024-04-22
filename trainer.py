from datetime import datetime
import os
import paddle
from player import AlphaGoPlayer
import numpy as np
from threading import Thread
from threading import Lock
import go_engine

lock = Lock()


class Trainer:
    def __init__(self, epochs=10, learning_rate=1e-3, batch_size=128, temp=1.0, n_playout=100, c_puct=5,
                 train_model_path='models/my_alpha_go.pdparams'):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.temp = temp
        self.n_playout = n_playout
        self.c_puct = c_puct
        self.train_model_path = train_model_path
        self.train_step = 0
        self.model_update_step = 0

        self.player = AlphaGoPlayer(train_model_path, c_puct, n_playout, is_selfplay=True)

        self.optimizer = paddle.optimizer.Momentum(learning_rate=learning_rate,
                                                   parameters=self.player.policy_value_net.parameters())

    def start(self, game):
        Thread(target=self._train, args=(game,), daemon=True).start()

    def _train(self, game):

        if os.path.exists(self.train_model_path):
            state_dict = paddle.load(self.train_model_path)
            self.player.policy_value_net.set_state_dict(state_dict)
            print('Loading model successfully！')
            game.info_display.push_text('{}  you wake up your own AI！'.format(
                datetime.now().strftime(r'%m-%d %H:%M:%S')), update=True)
        else:
            print('Model not found！')
            game.info_display.push_text('{}  you just create your own AI！'.format(
                datetime.now().strftime(r'%m-%d %H:%M:%S')), update=True)

        while True:
            if game.surface_state == 'play':
                break

            game.info_display.push_text('{} your AI start self-play！！'.format(
                datetime.now().strftime(r'%m-%d %H:%M:%S')), update=True)
            play_datas = self.self_play_one_game(game)
            if play_datas is not None:
                play_datas = self.get_equi_data(play_datas)

                self.update_network(game, play_datas)
                paddle.save(self.player.policy_value_net.state_dict(), self.train_model_path)
                self.model_update_step += 1
                print('save model ！')
                game.info_display.push_text('{}  AI stage{}！'.format(
                    datetime.now().strftime(r'%m-%d %H:%M:%S'), self.model_update_step), update=True)

    def self_play_one_game(self, game):
        states, mcts_probs, current_players = [], [], []

        while True:
            if game.surface_state == 'play':
                break

            move, move_probs = self.player.get_action(game, temp=self.temp, return_probs=True)
            states.append(game.train_game_state.get_board_state())
            mcts_probs.append(move_probs)
            current_players.append(game.train_game_state.turn())
            lock.acquire()
            if game.surface_state == 'train':
                game.train_step(move)
            lock.release()

            end, winner = game.train_game_state.game_ended(), game.train_game_state.winner()
            if end:
                print('{} wins！'.format('black' if winner == go_engine.BLACK else 'white'))
                game.info_display.push_text('{}  {} wins！'.format(
                    datetime.now().strftime(r'%m-%d %H:%M:%S'), 'black' if winner == go_engine.BLACK else 'white'), update=True)

                winners = np.zeros(len(current_players))
                if winner != -1:
                    winners[np.array(current_players) == winner] = 1.0
                    winners[np.array(current_players) != winner] = -1.0
                self.player.reset_player()
                game.train_game_state.reset()
                states = np.array(states)
                mcts_probs = np.array(mcts_probs)
                return zip(states, mcts_probs, winners)

    @staticmethod
    def get_equi_data(play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            board_size = state.shape[-1]
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                pass_move_prob = mcts_porb[-1]
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb[:-1].reshape(board_size, board_size)), i)
                extend_data.append((equi_state, np.append(np.flipud(equi_mcts_prob).flatten(), pass_move_prob), winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.append(np.flipud(equi_mcts_prob).flatten(), pass_move_prob), winner))
        return extend_data

    def update_network(self, game, play_datas):
        self.player.policy_value_net.train()
        for epoch in range(self.epochs):
            if game.surface_state == 'play':
                break

            np.random.shuffle(play_datas)
            for i in range(len(play_datas) // self.batch_size + 1):
                self.train_step += 1

                batch = play_datas[i * self.batch_size:(i + 1) * self.batch_size]
                if len(batch) == 0:
                    continue
                state_batch = paddle.to_tensor([data[0] for data in batch], dtype='float32')
                mcts_probs_batch = paddle.to_tensor([data[1] for data in batch], dtype='float32')
                winner_batch = paddle.to_tensor([data[2] for data in batch], dtype='float32')

                act_probs, value = self.player.policy_value_net(state_batch)
                ce_loss = paddle.nn.functional.cross_entropy(act_probs, mcts_probs_batch,
                                                             soft_label=True, use_softmax=False)
                mse_loss = paddle.nn.functional.mse_loss(value, winner_batch)
                loss = ce_loss + mse_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                print('{} Step:{} CELoss:{} MSELoss:{} Loss:{}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.train_step,
                    ce_loss.numpy(), mse_loss.numpy(), loss.numpy()))
        self.player.policy_value_net.eval()
