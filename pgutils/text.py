import pygame
from typing import List, Tuple, Union
import os

current_path = os.path.dirname(__file__)


def draw_text(surface: pygame.Surface,
              text: Union[str, pygame.Surface],
              pos: Union[List[str or float or int], Tuple[str or float or int]],
              font_size: int = 48,
              font_color: Union[Tuple[int], List[int]] = (255, 255, 255),
              font_path: str = current_path + "/../assets/fonts/msyh.ttc",
              next_bias: Union[Tuple[int or float], List[int or float]] = (0, 0)) -> Tuple:

    if isinstance(text, str):
        font = pygame.font.Font(font_path, font_size)
        text = font.render(text, True, font_color)

    pos = list(pos)
    if isinstance(pos[0], str):
        assert pos[0] == "center"
        pos[0] = (surface.get_width() - text.get_width()) / 2
    if isinstance(pos[1], str):
        assert pos[1] == "center"
        pos[1] = (surface.get_height() - text.get_height()) / 2

    surface.blit(text, pos)

    return pos[0] + next_bias[0], pos[1] + text.get_height() + next_bias[1]


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("test")

    draw_text(screen, 'Hello!', ["center", "center"])
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            pass
