"""
Game Selection Menu
====================

A visual menu displayed on app launch for selecting which game to train.

Features:
    - Game cards with icons and descriptions
    - Hover effects and animations
    - Keyboard and mouse navigation
    - Theme matching the dashboard aesthetic
"""

import pygame
from typing import Optional, Tuple, List, Dict, Any
import math


class GameCard:
    """A single game selection card."""
    
    def __init__(
        self,
        game_id: str,
        x: int,
        y: int,
        width: int,
        height: int,
        info: Dict[str, Any]
    ):
        self.game_id = game_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.info = info
        
        # Animation state
        self.hover = False
        self.hover_progress = 0.0  # 0 to 1
        self.selected = False
        
        # Colors from game info
        self.base_color = info.get('color', (100, 100, 100))
        
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def update(self, dt: float) -> None:
        """Update hover animation."""
        target = 1.0 if self.hover else 0.0
        self.hover_progress += (target - self.hover_progress) * min(1.0, dt * 10)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the game card."""
        # Calculate animated values
        scale = 1.0 + self.hover_progress * 0.05
        glow_alpha = int(self.hover_progress * 50)
        
        # Animated rect
        cx, cy = self.x + self.width // 2, self.y + self.height // 2
        w = int(self.width * scale)
        h = int(self.height * scale)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        
        # Draw glow effect when hovering
        if glow_alpha > 0:
            glow_rect = rect.inflate(20, 20)
            glow_color = (*self.base_color, glow_alpha)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=16)
            screen.blit(glow_surface, glow_rect)
        
        # Card background
        bg_color = (30, 35, 50) if not self.selected else (40, 50, 70)
        pygame.draw.rect(screen, bg_color, rect, border_radius=12)
        
        # Border
        border_color = self.base_color if self.hover else (50, 55, 70)
        border_width = 3 if self.hover else 1
        pygame.draw.rect(screen, border_color, rect, border_width, border_radius=12)
        
        # Icon
        icon = self.info.get('icon', '🎮')
        icon_font = pygame.font.Font(None, 64)
        try:
            icon_text = icon_font.render(icon, True, (255, 255, 255))
        except:
            icon_text = icon_font.render('?', True, (255, 255, 255))
        icon_rect = icon_text.get_rect(centerx=rect.centerx, top=rect.top + 20)
        screen.blit(icon_text, icon_rect)
        
        # Game name
        name = self.info.get('name', self.game_id.title())
        name_font = pygame.font.Font(None, 36)
        name_text = name_font.render(name, True, (255, 255, 255))
        name_rect = name_text.get_rect(centerx=rect.centerx, top=icon_rect.bottom + 15)
        screen.blit(name_text, name_rect)
        
        # Description
        desc = self.info.get('description', '')
        desc_font = pygame.font.Font(None, 22)
        desc_text = desc_font.render(desc, True, (150, 155, 170))
        desc_rect = desc_text.get_rect(centerx=rect.centerx, top=name_rect.bottom + 10)
        screen.blit(desc_text, desc_rect)
        
        # Difficulty badge
        difficulty = self.info.get('difficulty', 'Unknown')
        diff_font = pygame.font.Font(None, 20)
        diff_text = diff_font.render(difficulty, True, (200, 200, 200))
        diff_rect = diff_text.get_rect(centerx=rect.centerx, bottom=rect.bottom - 15)
        # Badge background
        badge_rect = diff_rect.inflate(16, 6)
        pygame.draw.rect(screen, (40, 45, 60), badge_rect, border_radius=10)
        screen.blit(diff_text, diff_rect)


class GameMenu:
    """
    Visual game selection menu.
    
    Displays available games as cards and allows selection
    via mouse or keyboard.
    
    Usage:
        >>> menu = GameMenu(screen_width=800, screen_height=600)
        >>> selected_game = menu.run(screen, clock)
        >>> if selected_game:
        >>>     # Start training with selected game
    """
    
    def __init__(self, screen_width: int = 800, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Background colors
        self.bg_color_top = (10, 12, 20)
        self.bg_color_bottom = (20, 25, 40)
        
        # Build game cards
        self.cards: List[GameCard] = []
        self.selected_index = 0
        self._build_cards()
        
        # Animation
        self.time = 0.0
        
        # Background stars
        self.stars = [(
            int(screen_width * (i * 0.618 % 1)),
            int(screen_height * (i * 0.382 % 1)),
            1 + (i % 3)
        ) for i in range(100)]
    
    def _build_cards(self) -> None:
        """Build game cards from registry."""
        # Import here to avoid circular imports
        from . import list_games, get_game_info
        
        games = list_games()
        num_games = len(games)
        
        if num_games == 0:
            return
        
        # Card dimensions
        card_width = 200
        card_height = 200
        card_spacing = 30
        
        # Calculate total width
        total_width = num_games * card_width + (num_games - 1) * card_spacing
        start_x = (self.screen_width - total_width) // 2
        y = (self.screen_height - card_height) // 2 + 30
        
        for i, game_id in enumerate(games):
            info = get_game_info(game_id) or {}
            x = start_x + i * (card_width + card_spacing)
            
            card = GameCard(game_id, x, y, card_width, card_height, info)
            self.cards.append(card)
        
        # Select first card
        if self.cards:
            self.cards[0].selected = True
    
    def run(self, screen: pygame.Surface, clock: pygame.time.Clock) -> Optional[str]:
        """
        Run the menu and return selected game.
        
        Args:
            screen: Pygame surface to draw on
            clock: Pygame clock for timing
            
        Returns:
            Selected game ID, or None if cancelled
        """
        running = True
        
        while running:
            dt = clock.tick(60) / 1000.0
            self.time += dt
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_LEFT:
                        self._select_prev()
                    elif event.key == pygame.K_RIGHT:
                        self._select_next()
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if self.cards:
                            return self.cards[self.selected_index].game_id
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_mouse_motion(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        selected = self._handle_click(event.pos)
                        if selected:
                            return selected
            
            # Update
            for card in self.cards:
                card.update(dt)
            
            # Draw
            self._draw(screen)
            pygame.display.flip()
        
        return None
    
    def _select_prev(self) -> None:
        """Select previous card."""
        if not self.cards:
            return
        self.cards[self.selected_index].selected = False
        self.selected_index = (self.selected_index - 1) % len(self.cards)
        self.cards[self.selected_index].selected = True
    
    def _select_next(self) -> None:
        """Select next card."""
        if not self.cards:
            return
        self.cards[self.selected_index].selected = False
        self.selected_index = (self.selected_index + 1) % len(self.cards)
        self.cards[self.selected_index].selected = True
    
    def _handle_mouse_motion(self, pos: Tuple[int, int]) -> None:
        """Handle mouse hover."""
        for i, card in enumerate(self.cards):
            card.hover = card.rect.collidepoint(pos)
            if card.hover and self.selected_index != i:
                self.cards[self.selected_index].selected = False
                self.selected_index = i
                card.selected = True
    
    def _handle_click(self, pos: Tuple[int, int]) -> Optional[str]:
        """Handle mouse click. Returns game ID if card clicked."""
        for card in self.cards:
            if card.rect.collidepoint(pos):
                return card.game_id
        return None
    
    def _draw(self, screen: pygame.Surface) -> None:
        """Draw the menu."""
        # Gradient background
        for y in range(self.screen_height):
            t = y / self.screen_height
            color = tuple(int(self.bg_color_top[i] + t * (self.bg_color_bottom[i] - self.bg_color_top[i])) for i in range(3))
            pygame.draw.line(screen, color, (0, y), (self.screen_width, y))
        
        # Animated stars
        for x, y, size in self.stars:
            alpha = int(100 + 50 * math.sin(self.time * 2 + x * 0.01))
            star_color = (alpha, alpha, alpha + 20)
            pygame.draw.circle(screen, star_color, (x, y), size)
        
        # Title
        title_font = pygame.font.Font(None, 72)
        title_text = title_font.render("Neural Network AI", True, (255, 255, 255))
        title_rect = title_text.get_rect(centerx=self.screen_width // 2, top=50)
        screen.blit(title_text, title_rect)
        
        # Subtitle
        sub_font = pygame.font.Font(None, 32)
        sub_text = sub_font.render("Select a game to train", True, (150, 155, 170))
        sub_rect = sub_text.get_rect(centerx=self.screen_width // 2, top=title_rect.bottom + 10)
        screen.blit(sub_text, sub_rect)
        
        # Draw cards
        for card in self.cards:
            card.draw(screen)
        
        # Instructions
        inst_font = pygame.font.Font(None, 24)
        inst_text = inst_font.render("← → to select  |  Enter to start  |  Esc to quit", True, (100, 105, 120))
        inst_rect = inst_text.get_rect(centerx=self.screen_width // 2, bottom=self.screen_height - 30)
        screen.blit(inst_text, inst_rect)


if __name__ == "__main__":
    # Test the menu
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Game Selection - Test")
    clock = pygame.time.Clock()
    
    menu = GameMenu(800, 600)
    selected = menu.run(screen, clock)
    
    print(f"Selected game: {selected}")
    pygame.quit()

