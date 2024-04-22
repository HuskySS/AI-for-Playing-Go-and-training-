import pygame


class CtBase:
    def __init__(self):
        self.active = False

    def enable(self):
        self.active = True

    def disable(self):
        self.active = False

    def update(self, event: pygame.event) -> ...:

        raise NotImplementedError
