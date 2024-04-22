import pygame
import os
import copy
from pgutils.text import draw_text
from pgutils.position import pos_in_surface
from pgutils.pgcontrols.ctbase import CtBase
from typing import Tuple, List, Union, Callable, Optional

current_path = os.path.dirname(__file__)


class Button(CtBase):
    def __init__(self, surface: pygame.Surface,
                 text: str,
                 pos: Union[Tuple[str or int], List[str or int]],
                 call_function: Optional[Callable] = None,
                 click_sound: Union[str, pygame.mixer.Sound] = current_path + "/../../assets/audios/Button.wav",
                 font_path: str = current_path + "/../../assets/fonts/msyh.ttc",
                 font_size: int = 14,
                 size: Union[Tuple[int], List[int]] = (87, 27),
                 text_color: Union[Tuple[int], List[int]] = (0, 0, 0),
                 up_color: Union[Tuple[int], List[int]] = (225, 225, 225),
                 down_color: Union[Tuple[int], List[int]] = (190, 190, 190),
                 outer_edge_color: Union[Tuple[int], List[int]] = (240, 240, 240),
                 inner_edge_color: Union[Tuple[int], List[int]] = (173, 173, 173)):

        super(Button, self).__init__()

        pos = copy.copy(list(pos))
        if isinstance(pos[0], str):
            assert pos[0] == "center"
            pos[0] = (surface.get_width() - size[0]) // 2
        if isinstance(pos[1], str):
            assert pos[1] == "center"
            pos[1] = (surface.get_height() - size[1]) // 2
        if isinstance(click_sound, str):
            click_sound = pygame.mixer.Sound(click_sound)

        self.button_surface = surface.subsurface(pos[0], pos[1], size[0], size[1])
        self.outer_rect = 0, 0, size[0], size[1]
        self.inner_rect = self.outer_rect[0] + 1, self.outer_rect[1] + 1, self.outer_rect[2] - 2, self.outer_rect[3] - 2

        self.font = pygame.font.Font(font_path, font_size)
        self.text = self.font.render(text, True, text_color)
        self.text_color = text_color
        self.size = size
        self.call_function = call_function
        self.click_sound = click_sound
        self.up_color = up_color
        self.down_color = down_color
        self.outer_edge_color = outer_edge_color
        self.inner_edge_color = inner_edge_color
        self.is_down = False

    def draw_up(self):
        self.is_down = False
        self.draw(self.up_color)

    def draw_down(self):
        self.is_down = True
        self.draw(self.down_color)

    def draw(self, base_color: Union[Tuple[int], List[int]]):
        self.button_surface.fill(base_color)
        pygame.draw.rect(self.button_surface, self.outer_edge_color, self.outer_rect, width=1)
        pygame.draw.rect(self.button_surface, self.inner_edge_color, self.inner_rect, width=1)
        draw_text(self.button_surface, self.text, ["center", "center"])

    def set_text(self, text: str, draw_update: bool = True):
        self.text = self.font.render(text, True, self.text_color)
        if draw_update:
            self.draw_up()

    def enable(self):
        self.active = True
        self.draw_up()

    def disable(self):
        self.active = False
        self.draw_down()

    def update(self, event: pygame.event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if pos_in_surface(event.pos, self.button_surface):
                self.draw_down()
                self.is_down = True
        elif event.type == pygame.MOUSEMOTION:
            if not pos_in_surface(event.pos, self.button_surface) and self.is_down:
                self.draw_up()
                self.is_down = False
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if pos_in_surface(event.pos, self.button_surface) and self.is_down:
                self.draw_up()
                self.click_sound.play()
                if self.call_function is not None:
                    self.call_function()


