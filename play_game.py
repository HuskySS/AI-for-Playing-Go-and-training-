from game_engine import GameEngine
import pygame
import sys


if __name__ == '__main__':
    game = GameEngine()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            else:
                game.event_control(event)
        game.take_action()
        game.music_control()
        pygame.display.update()
