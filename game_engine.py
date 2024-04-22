import pygame
import sys
import copy
import go_engine
from go_engine import GoEngine
from pgutils.manager import Manager
from pgutils.pgcontrols.button import Button
from pgutils.text import draw_text
from pgutils.pgtools.information_display import InformationDisplay
from pgutils.position import pos_in_surface
from player import *
import os
from typing import List, Tuple, Callable, Union
from trainer import Trainer

SCREEN_SIZE = 1.8
SCREENWIDTH = int(SCREEN_SIZE * 600)
SCREENHEIGHT = int(SCREEN_SIZE * 400)
BGCOLOR = (53, 107, 162)
BOARDCOLOR = (204, 85, 17)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
MARKCOLOR = (0, 200, 200)

pygame.init()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('GO Game')
loading_font = pygame.font.Font('assets/fonts/msyh.ttc', 72)
loading_text_render = loading_font.render('Loading...', True, WHITE)
SCREEN.blit(loading_text_render, ((SCREEN.get_width() - loading_text_render.get_width()) / 2,
                                  (SCREEN.get_height() - loading_text_render.get_height()) / 2))
pygame.display.update()
pygame.mixer.init()
IMAGES = {'black': (
    pygame.image.load('assets/pictures/B.png').convert_alpha(),
    pygame.image.load('assets/pictures/B-9.png').convert_alpha(),
    pygame.image.load('assets/pictures/B-13.png').convert_alpha(),
    pygame.image.load('assets/pictures/B-19.png').convert_alpha()
), 'white': (
    pygame.image.load('assets/pictures/W.png').convert_alpha(),
    pygame.image.load('assets/pictures/W-9.png').convert_alpha(),
    pygame.image.load('assets/pictures/W-13.png').convert_alpha(),
    pygame.image.load('assets/pictures/W-19.png').convert_alpha()
)}
SOUNDS = {
    'stone': pygame.mixer.Sound('assets/audios/Stone.wav'),
    'button': pygame.mixer.Sound('assets/audios/Button.wav')
}
MUSICS = [[os.path.splitext(music)[0], pygame.mixer.Sound('assets/musics/' + music)]
          for music in os.listdir('assets/musics')]


class GameEngine:
    def __init__(self, board_size: int = 9,
                 komi=7.5,
                 record_step: int = 4,
                 state_format: str = "separated",
                 record_last: bool = True):

        assert board_size in [9, 13, 19]
        assert state_format in ["separated", "merged"]

        self.board_size = board_size
        self.komi = komi
        self.record_step = record_step
        self.state_format = state_format
        self.record_last = record_last

        self.game_state = GoEngine(board_size=board_size, komi=komi, record_step=record_step,
                                   state_format=state_format, record_last=record_last)
        self.trainer = Trainer()
        self.train_game_state = None

        self.manager = Manager()
        self.play_state = False
        self.surface_state = 'play'
        self.train_state = False

        self.music_control_name = ['shuffle', 'in order', 'single repeat', 'turn off']
        self.black_player = HumanPlayer()
        self.white_player = HumanPlayer()
        self.black_player_id = 0
        self.white_player_id = 0
        self.music_id = 0
        self.music_control_id = 0

        SCREEN.fill(BGCOLOR)
        self.board_surface = SCREEN.subsurface((0, 0, SCREENHEIGHT, SCREENHEIGHT))
        self.speed_surface = SCREEN.subsurface((SCREENHEIGHT, 0, 3, SCREENHEIGHT))
        self.taiji_surface = SCREEN.subsurface((SCREENHEIGHT + self.speed_surface.get_width(), 0,
                                                SCREENWIDTH - SCREENHEIGHT - self.speed_surface.get_width(),
                                                SCREENHEIGHT / 3.5))
        self.pmc_surface = SCREEN.subsurface((SCREENHEIGHT + self.speed_surface.get_width(), SCREENHEIGHT / 3.5,
                                              SCREENWIDTH - SCREENHEIGHT - self.speed_surface.get_width(),
                                              SCREENHEIGHT / 5))
        self.operate_surface = SCREEN.subsurface((SCREENHEIGHT + self.speed_surface.get_width(),
                                                  self.taiji_surface.get_height() + self.pmc_surface.get_height(),
                                                  SCREENWIDTH - SCREENHEIGHT - self.speed_surface.get_width(),
                                                  SCREENHEIGHT * (1 - 1 / 3.5 - 1 / 5)))

        self.info_display = InformationDisplay(self.operate_surface, max_show=12, display_pos=[15, 15],
                                               display_size=[self.operate_surface.get_width() - 30, 230])
        self.manager.tool_register(self.info_display)

        pmc_button_texts = [self.black_player.name, self.white_player.name,
                            MUSICS[self.music_id][0], self.music_control_name[self.music_control_id]]
        pmc_button_call_functions = [self.fct_for_black_player, self.fct_for_white_player,
                                     self.fct_for_music_choose, self.fct_for_music_control]
        self.pmc_buttons = self.create_buttons(self.pmc_surface, pmc_button_texts, pmc_button_call_functions,
                                               [22 * SCREEN_SIZE + 120, self.pmc_surface.get_height() / 20 + 4],
                                               18 * SCREEN_SIZE, up_color=[202, 171, 125], down_color=[186, 146, 86],
                                               outer_edge_color=[255, 255, 214], size=[160, 27], font_size=16,
                                               inner_edge_color=[247, 207, 181], text_color=[253, 253, 19])
        operate_play_button_texts = ['start', 'pass', 'regret', 'restart', ('13' if self.board_size == 9 else '9') + 'rows',
                                     ('13' if self.board_size == 19 else '19') + 'rows', 'AI training', 'exit']
        operate_play_button_call_functions = [self.fct_for_play_game, self.fct_for_pass, self.fct_for_regret,
                                              self.fct_for_restart, self.fct_for_new_game_1, self.fct_for_new_game_2,
                                              self.fct_for_train_alphago, self.fct_for_exit]
        self.operate_play_buttons = self.create_buttons(self.operate_surface, operate_play_button_texts,
                                                        operate_play_button_call_functions,
                                                        ['center', self.operate_surface.get_height() / 20],
                                                        24 * SCREEN_SIZE, size=[120, 27])
        operate_train_button_texts = ['Start training', 'back']
        operate_train_button_call_functions = [self.fct_for_train, self.fct_for_back]
        self.operate_train_buttons = \
            self.create_buttons(self.operate_surface, operate_train_button_texts, operate_train_button_call_functions,
                                ['center', (self.operate_surface.get_height() +
                                            (self.info_display.display_size[1] +
                                            self.info_display.display_pos[1]) * 4) / 5],
                                26 * SCREEN_SIZE, size=[120, 27])

        self.manager.control_register(self.pmc_buttons)
        self.manager.control_register(self.operate_play_buttons)
        self.manager.control_register(self.operate_train_buttons)

        self.block_size = int(SCREEN_SIZE * 360 / (self.board_size - 1))
        if self.board_size == 9:
            self.piece_size = IMAGES['black'][1].get_size()
        elif self.board_size == 13:
            self.piece_size = IMAGES['black'][2].get_size()
        else:
            self.piece_size = IMAGES['black'][3].get_size()

        self.draw_board()
        self.draw_taiji()
        self.draw_pmc()
        self.draw_operate()

        if not pygame.mixer.get_busy():
            MUSICS[self.music_id][1].play()

        pygame.display.update()

    def draw_board(self):
        self.board_surface.fill(BOARDCOLOR)
        rect_pos = (int(SCREEN_SIZE * 20), int(SCREEN_SIZE * 20), int(SCREEN_SIZE * 360), int(SCREEN_SIZE * 360))
        pygame.draw.rect(self.board_surface, BLACK, rect_pos, 3)
        for i in range(self.board_size - 2):
            pygame.draw.line(self.board_surface, BLACK,
                             (SCREEN_SIZE * 20, SCREEN_SIZE * 20 + (i + 1) * self.block_size),
                             (SCREEN_SIZE * 380, SCREEN_SIZE * 20 + (i + 1) * self.block_size), 2)
            pygame.draw.line(self.board_surface, BLACK,
                             (SCREEN_SIZE * 20 + (i + 1) * self.block_size, SCREEN_SIZE * 20),
                             (SCREEN_SIZE * 20 + (i + 1) * self.block_size, SCREEN_SIZE * 380), 2)
        if self.board_size == 9:
            position_loc = [2, 4, 6]
        elif self.board_size == 13:
            position_loc = [3, 6, 9]
        else:
            position_loc = [3, 9, 15]
        positions = [[SCREEN_SIZE * 20 + 1 + self.block_size * i, SCREEN_SIZE * 20 + 1 + self.block_size * j]
                     for i in position_loc for j in position_loc]
        for pos in positions:
            pygame.draw.circle(self.board_surface, BLACK, pos, 5, 0)

    def draw_taiji(self):
        game_state = self.game_state if self.surface_state == 'play' else self.train_game_state

        black_pos = (self.taiji_surface.get_width() - IMAGES['black'][0].get_width()) / 2, \
                    (self.taiji_surface.get_height() - IMAGES['black'][0].get_height()) / 2
        white_pos = black_pos[0] + 44, black_pos[1]
        self.taiji_surface.fill(BGCOLOR)
        if not self.play_state and self.surface_state == 'play' or \
                not self.train_state and self.surface_state == 'train':

            self.taiji_surface.blit(IMAGES['black'][0], black_pos)
            self.taiji_surface.blit(IMAGES['white'][0], white_pos)
        else:
            if game_state.turn() == go_engine.BLACK:
                self.taiji_surface.blit(IMAGES['black'][0], black_pos)
            elif game_state.turn() == go_engine.WHITE:
                self.taiji_surface.blit(IMAGES['white'][0], white_pos)

    def draw_pmc(self):
        self.pmc_surface.fill(BGCOLOR)
        texts = ['Black：', 'White：', 'BGM：', 'Control：']
        pos_next = [22 * SCREEN_SIZE, self.pmc_surface.get_height() / 20]
        for text in texts:
            pos_next = draw_text(self.pmc_surface, text, pos_next, font_size=24)
        for button in self.pmc_buttons:
            button.enable()

    def draw_operate(self):
        self.operate_surface.fill(BGCOLOR)
        if self.surface_state == 'play':
            self.operate_surface.fill(BGCOLOR)
            for button in self.operate_play_buttons:
                button.enable()
        elif self.surface_state == 'train':
            self.info_display.enable()
            for button in self.operate_train_buttons:
                button.enable()
        if self.board_size in [13, 19]:
            self.operate_play_buttons[6].disable()

    def draw_pieces(self) -> None:
        game_state = self.game_state if self.surface_state == 'play' else self.train_game_state

        for i in range(self.board_size):
            for j in range(self.board_size):
                pos = (SCREEN_SIZE * 22 + self.block_size * j - self.piece_size[1] / 2,
                       SCREEN_SIZE * 19 + self.block_size * i - self.piece_size[0] / 2)
                if game_state.current_state[go_engine.BLACK][i, j] == 1:
                    if self.board_size == 9:
                        self.board_surface.blit(IMAGES['black'][1], pos)
                    elif self.board_size == 13:
                        self.board_surface.blit(IMAGES['black'][2], pos)
                    else:
                        self.board_surface.blit(IMAGES['black'][3], pos)
                elif game_state.current_state[go_engine.WHITE][i, j] == 1:
                    if self.board_size == 9:
                        self.board_surface.blit(IMAGES['white'][1], pos)
                    elif self.board_size == 13:
                        self.board_surface.blit(IMAGES['white'][2], pos)
                    else:
                        self.board_surface.blit(IMAGES['white'][3], pos)

    def draw_mark(self, action):
        if action != self.board_size ** 2:
            game_state = self.game_state if self.surface_state == 'play' else self.train_game_state

            row = action // self.board_size
            col = action % self.board_size
            if game_state.turn() == go_engine.WHITE:
                if self.board_size == 9:
                    pos = (SCREEN_SIZE * 19 + col * self.block_size, SCREEN_SIZE * 22 + row * self.block_size)
                elif self.board_size == 13:
                    pos = (SCREEN_SIZE * 20 + col * self.block_size, SCREEN_SIZE * 21 + row * self.block_size)
                else:
                    pos = (SCREEN_SIZE * 21 + col * self.block_size, SCREEN_SIZE * 20 + row * self.block_size)
            else:
                if self.board_size == 9:
                    pos = (SCREEN_SIZE * 19 + col * self.block_size, SCREEN_SIZE * 20 + row * self.block_size)
                elif self.board_size == 13:
                    pos = (SCREEN_SIZE * 20 + col * self.block_size, SCREEN_SIZE * 20 + row * self.block_size)
                else:
                    pos = (SCREEN_SIZE * 21 + col * self.block_size, SCREEN_SIZE * 19 + row * self.block_size)
            pygame.draw.circle(self.board_surface, MARKCOLOR, pos, self.piece_size[0] / 2 + 2 * SCREEN_SIZE, 2)

    def play_step(self, action):

        self.game_state.step(action)
        if action != self.board_size ** 2 and action is not None:
            self.draw_board()
            self.draw_pieces()
            self.draw_mark(action)
            SOUNDS['stone'].play()
        self.draw_taiji()
        if self.game_state.done:
            self.draw_over()
            self.play_state = False
            self.operate_play_buttons[0].set_text('start game')
            self.draw_taiji()

    def train_step(self, action):
        self.train_game_state.step(action)
        if action != self.board_size ** 2 and action is not None:
            self.draw_board()
            self.draw_pieces()
            self.draw_mark(action)
        self.draw_taiji()

    def take_action(self):
        if self.play_state and self.game_state.turn() == go_engine.BLACK and \
                not isinstance(self.black_player, HumanPlayer):
            self.black_player.play(self)
        elif self.play_state and self.game_state.turn() == go_engine.WHITE and \
                not isinstance(self.white_player, HumanPlayer):
            self.white_player.play(self)

        if self.play_state and self.game_state.turn() == go_engine.BLACK and self.black_player.action is not None:
            self.play_step(self.black_player.action)
            self.black_player.action = None
            self.white_player.allow = True
        elif self.play_state and self.game_state.turn() == go_engine.WHITE and self.white_player.action is not None:
            self.play_step(self.white_player.action)
            self.white_player.action = None
            self.black_player.allow = True

        if self.black_player.speed is not None:
            self.draw_speed(self.black_player.speed[0], self.black_player.speed[1])
            self.black_player.speed = None
        elif self.white_player.speed is not None:
            self.draw_speed(self.white_player.speed[0], self.white_player.speed[1])
            self.white_player.speed = None

        if self.surface_state == 'train':
            self.manager.tool_update()

    def event_control(self, event: pygame.event.Event):
        next_player = self.next_player()
        if self.play_state and isinstance(next_player, HumanPlayer):
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and \
                    pos_in_surface(event.pos, self.board_surface):
                action = self.mouse_pos_to_action(event.pos)
                if self.game_state.action_valid(action):
                    if self.game_state.turn() == go_engine.BLACK:
                        self.black_player.action = action
                    else:
                        self.white_player.action = action
        self.manager.control_update(event)

    def game_state_simulator(self, train=False) -> GoEngine:
        game_state = GoEngine(board_size=self.board_size, komi=self.komi, record_step=self.record_step,
                              state_format=self.state_format, record_last=self.record_last)

        if not train:
            game_state.current_state = np.copy(self.game_state.current_state)
            game_state.board_state = np.copy(self.game_state.board_state)
            game_state.board_state_history = copy.copy(self.game_state.board_state_history)
            game_state.action_history = copy.copy(self.game_state.action_history)
            game_state.done = self.game_state.done
        else:
            game_state.current_state = np.copy(self.train_game_state.current_state)
            game_state.board_state = np.copy(self.train_game_state.board_state)
            game_state.board_state_history = copy.copy(self.train_game_state.board_state_history)
            game_state.action_history = copy.copy(self.train_game_state.action_history)
            game_state.done = self.train_game_state.done
        return game_state

    def mouse_pos_to_action(self, mouse_pos):
        if 0 < mouse_pos[0] < SCREENHEIGHT and 0 < mouse_pos[1] < SCREENHEIGHT:
            row = round((mouse_pos[1] - SCREEN_SIZE * 20) / self.block_size)
            if row < 0:
                row = 0
            elif row > self.board_size - 1:
                row = self.board_size - 1
            col = round((mouse_pos[0] - SCREEN_SIZE * 20) / self.block_size)
            if col < 0:
                col = 0
            elif col > self.board_size - 1:
                col = self.board_size - 1
            return row * self.board_size + col

    def draw_over(self):
        black_areas, white_areas = self.game_state.areas()
        over_text_1 = 'negotiation'
        over_text_2 = 'black{}stones white{}stones'.format(black_areas, white_areas)
        area_difference = (black_areas - white_areas - self.komi) / 2
        if area_difference == 0:
            over_text_3 = 'draw'
        elif area_difference > 0:
            over_text_3 = 'black wins {} stones'.format(area_difference)
        else:
            over_text_3 = 'white wins {} stones '.format(-area_difference)
        over_screen = pygame.Surface((320, 170), pygame.SRCALPHA)
        over_screen.fill((57, 44, 33, 100))
        next_pos = draw_text(over_screen, over_text_1, ['center', over_screen.get_height() / 6],
                             font_size=26, font_color=[220, 220, 220])
        next_pos = draw_text(over_screen, over_text_2, ['center', next_pos[1]],
                             font_size=26, font_color=[220, 220, 220])
        draw_text(over_screen, over_text_3, ['center', next_pos[1]], font_size=26, font_color=[220, 220, 220])
        self.board_surface.blit(over_screen,
                                ((self.board_surface.get_width() - over_screen.get_width()) / 2,
                                 (self.board_surface.get_height() - over_screen.get_height()) / 2))

    def draw_speed(self, count, total):
        self.speed_surface.fill(BGCOLOR)
        sub_speed_area = self.speed_surface.subsurface((0, SCREENHEIGHT - round(count / total * SCREENHEIGHT),
                                                        self.speed_surface.get_width(),
                                                        round(count / total * SCREENHEIGHT)))
        sub_speed_area.fill((15, 255, 255))

    @staticmethod
    def create_buttons(surface: pygame.Surface,
                       button_texts: List[str],
                       call_functions: List[Callable],
                       first_pos: List[int or float],
                       button_margin: Union[int, float],
                       font_size: int = 14,
                       size: Union[Tuple[int], List[int]] = (87, 27),
                       text_color: Union[Tuple[int], List[int]] = (0, 0, 0),
                       up_color: Union[Tuple[int], List[int]] = (225, 225, 225),
                       down_color: Union[Tuple[int], List[int]] = (190, 190, 190),
                       outer_edge_color: Union[Tuple[int], List[int]] = (240, 240, 240),
                       inner_edge_color: Union[Tuple[int], List[int]] = (173, 173, 173)
                       ) -> List[Button]:

        assert len(button_texts) == len(call_functions)

        buttons = []

        pos_next = copy.copy(first_pos)
        for btn_text, call_fct in zip(button_texts, call_functions):
            button = Button(surface, btn_text, pos_next, call_fct, size=size, font_size=font_size,
                            text_color=text_color, up_color=up_color, down_color=down_color,
                            outer_edge_color=outer_edge_color, inner_edge_color=inner_edge_color)
            buttons.append(button)
            pos_next[1] += button_margin
        return buttons

    def music_control(self):
        if not pygame.mixer.get_busy() and self.music_control_id != 3:
            if self.music_control_id == 0:
                rand_int = np.random.randint(len(MUSICS))
                if len(MUSICS) > 1:
                    while rand_int == self.music_id:
                        rand_int = np.random.randint(len(MUSICS))
                self.music_id = rand_int
                MUSICS[self.music_id][1].play()
            elif self.music_control_id == 1:
                self.music_id += 1
                self.music_id %= len(MUSICS)
                MUSICS[self.music_id][1].play()
            elif self.music_control_id == 2:
                MUSICS[self.music_id][1].play()
            self.pmc_buttons[2].set_text(MUSICS[self.music_id][0])
            self.pmc_buttons[2].draw_up()
        elif pygame.mixer.get_busy() and self.music_control_id == 3:
            MUSICS[self.music_id][1].stop()

    def next_player(self):
        if self.game_state.turn() == go_engine.BLACK:
            return self.black_player
        else:
            return self.white_player

    def fct_for_black_player(self):
        self.play_state = False
        self.black_player.valid = False
        self.operate_play_buttons[0].set_text('game start')

        if self.game_state.turn() == go_engine.BLACK:
            if isinstance(self.black_player, MCTSPlayer) or isinstance(self.black_player, AlphaGoPlayer):
                self.draw_speed(0, 1)

        self.black_player_id += 1
        player_num = 11 if self.board_size == 9 else 2
        self.black_player_id %= player_num

        self.black_player = self.create_player(self.black_player_id)
        self.pmc_buttons[0].set_text(self.black_player.name)

    def fct_for_white_player(self):
        self.play_state = False
        self.white_player.valid = False
        self.operate_play_buttons[0].set_text('game start')

        if self.game_state.turn() == go_engine.WHITE:
            if isinstance(self.white_player, MCTSPlayer) or isinstance(self.white_player, AlphaGoPlayer):
                self.draw_speed(0, 1)

        self.white_player_id += 1
        player_num = 11 if self.board_size == 9 else 2
        self.white_player_id %= player_num

        self.white_player = self.create_player(self.white_player_id)
        self.pmc_buttons[1].set_text(self.white_player.name)

    @staticmethod
    def create_player(player_id: int) -> Player:
        if player_id == 0:
            player = HumanPlayer()
        elif player_id == 1:
            player = RandomPlayer()
        elif player_id in [2, 3, 4, 5, 6]:
            player = MCTSPlayer(n_playout=400 * (2 ** (player_id - 2)))
        elif player_id == 7:
            player = PolicyNetPlayer(model_path='models/alpha_go.pdparams')
        elif player_id == 8:
            player = ValueNetPlayer(model_path='models/alpha_go.pdparams')
        elif player_id == 9:
            player = AlphaGoPlayer(model_path='models/alpha_go.pdparams')
        elif player_id == 10:
            player = AlphaGoPlayer(model_path='models/my_alpha_go.pdparams')
        else:
            player = Player()
        return player

    def fct_for_music_choose(self):
        MUSICS[self.music_id][1].stop()
        if self.music_control_id == 0:
            rand_int = np.random.randint(len(MUSICS))
            if len(MUSICS) > 1:
                while rand_int == self.music_id:
                    rand_int = np.random.randint(len(MUSICS))
            self.music_id = rand_int
        else:
            self.music_id += 1
            self.music_id %= len(MUSICS)
        self.pmc_buttons[2].set_text(MUSICS[self.music_id][0])
        MUSICS[self.music_id][1].play()

    def fct_for_music_control(self):
        self.music_control_id += 1
        self.music_control_id %= len(self.music_control_name)
        self.pmc_buttons[3].set_text(self.music_control_name[self.music_control_id])
        if self.music_control_id == 0:
            MUSICS[self.music_id][1].play()

    def fct_for_play_game(self):
        if self.play_state:
            self.operate_play_buttons[0].set_text('game start')
            self.play_state = False
            self.next_player().valid = False
        else:
            if self.game_state.done:
                self.game_state.reset()
                self.draw_board()
                self.black_player.allow = True
            else:
                self.next_player().valid = True
            self.operate_play_buttons[0].set_text('pause')
            self.play_state = True
        self.draw_taiji()

    def fct_for_pass(self):
        if self.play_state:
            next_player = self.next_player()
            if isinstance(next_player, HumanPlayer):
                if self.game_state.turn() == go_engine.BLACK:
                    self.black_player.action = self.board_size ** 2
                else:
                    self.white_player.action = self.board_size ** 2

    def fct_for_regret(self):
        if self.play_state:
            next_player = self.next_player()
            if isinstance(next_player, HumanPlayer):
                if len(self.game_state.board_state_history) > 2:
                    self.game_state.current_state = self.game_state.board_state_history[-3]
                    self.game_state.board_state_history = self.game_state.board_state_history[:-2]
                    action = self.game_state.action_history[-3]
                    self.game_state.action_history = self.game_state.action_history[:-2]
                    self.draw_board()
                    self.draw_pieces()
                    self.draw_mark(action)
                    self.draw_taiji()
                elif len(self.game_state.board_state_history) == 2:
                    self.game_state.reset()
                    self.draw_board()
                    self.draw_taiji()

    def fct_for_restart(self):
        self.play_state = True
        self.operate_play_buttons[0].set_text('pause')

        self.black_player.valid = False
        self.white_player.valid = False
        self.black_player = self.create_player(self.black_player_id)
        self.white_player = self.create_player(self.white_player_id)

        self.game_state.reset()
        self.draw_board()
        self.draw_taiji()

    def fct_for_new_game_1(self):
        self.black_player.valid = False
        self.white_player.valid = False
        music_id = self.music_id
        music_control_id = self.music_control_id
        new_game_size = 13 if self.board_size == 9 else 9
        self.__init__(new_game_size, komi=self.komi, record_step=self.record_step, state_format=self.state_format,
                      record_last=self.record_last)
        self.music_id = music_id
        self.music_control_id = music_control_id
        self.pmc_buttons[2].set_text(MUSICS[self.music_id][0])
        self.pmc_buttons[3].set_text(self.music_control_name[self.music_control_id])

    def fct_for_new_game_2(self):
        self.black_player.valid = False
        self.white_player.valid = False

        music_id = self.music_id
        music_control_id = self.music_control_id
        new_game_size = 13 if self.board_size == 19 else 19
        self.__init__(new_game_size, komi=self.komi, record_step=self.record_step, state_format=self.state_format,
                      record_last=self.record_last)
        self.music_id = music_id
        self.music_control_id = music_control_id
        self.pmc_buttons[2].set_text(MUSICS[self.music_id][0])
        self.pmc_buttons[3].set_text(self.music_control_name[self.music_control_id])

    def fct_for_train_alphago(self):
        self.surface_state = 'train'
        self.play_state = False
        self.black_player.valid = False
        self.white_player.valid = False
        self.train_game_state = GoEngine(board_size=self.board_size, komi=self.komi, record_step=self.record_step,
                                         state_format=self.state_format, record_last=self.record_last)
        self.draw_board()

        self.pmc_buttons[0].set_text('training')
        self.pmc_buttons[0].disable()
        self.pmc_buttons[1].set_text('training')
        self.pmc_buttons[1].disable()
        for button in self.operate_play_buttons:
            button.disable()

        self.draw_taiji()
        self.draw_operate()
        self.info_display.enable()

    @staticmethod
    def fct_for_exit():
        sys.exit()

    def fct_for_train(self):
        if not self.train_state:
            self.train_state = True
            self.trainer.start(self)

    def fct_for_back(self):
        self.surface_state = 'play'
        self.train_state = False
        self.black_player.valid = True
        self.white_player.valid = True
        self.info_display.disable()

        self.draw_board()
        self.draw_pieces()
        self.draw_taiji()
        if len(self.game_state.action_history) > 0:
            self.draw_mark(self.game_state.action_history[-1])

        self.operate_play_buttons[0].set_text('start game')
        for button in self.operate_train_buttons:
            button.disable()
        self.pmc_buttons[0].set_text(self.black_player.name)
        self.pmc_buttons[0].enable()
        self.pmc_buttons[1].set_text(self.white_player.name)
        self.pmc_buttons[1].enable()

        self.draw_operate()


if __name__ == '__main__':
    game = GameEngine()
    while True:
        for evt in pygame.event.get():
            game.event_control(evt)
        game.take_action()
        game.music_control()
        pygame.display.update()
