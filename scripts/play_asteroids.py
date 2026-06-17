"""Run Asteroids with human controls for local manual testing."""

from __future__ import annotations

import pygame

from config import Config
from src.game.asteroids import Asteroids


def main() -> None:
    """Launch a small pygame loop for manual Asteroids play."""
    pygame.init()
    config = Config()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroids - Human Play Test")
    clock = pygame.time.Clock()

    game = Asteroids(config)
    game.show_controls = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_h:
                    game.show_controls = not game.show_controls

        pressed = pygame.key.get_pressed()
        keys = {
            pygame.K_LEFT: pressed[pygame.K_LEFT],
            pygame.K_RIGHT: pressed[pygame.K_RIGHT],
            pygame.K_UP: pressed[pygame.K_UP],
            pygame.K_DOWN: pressed[pygame.K_DOWN],
            pygame.K_SPACE: pressed[pygame.K_SPACE],
            pygame.K_w: pressed[pygame.K_w],
            pygame.K_a: pressed[pygame.K_a],
            pygame.K_s: pressed[pygame.K_s],
            pygame.K_d: pressed[pygame.K_d],
        }

        _state, _reward, done, _info = game.step_human(keys)
        if done:
            pygame.time.wait(2000)
            game.reset()

        game.render(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
