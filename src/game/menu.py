"""
Game Selection Menu
====================

A visual menu displayed on app launch for selecting which game to train.

Features:
    - Game cards with custom drawn icons
    - Hover effects and animations
    - Keyboard and mouse navigation
    - Styled action buttons
"""

import pygame
from typing import Optional, Tuple, List, Dict, Any
import math


def draw_breakout_icon(surface: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int]) -> None:
    """Draw a Breakout-style icon."""
    cx, cy = rect.centerx, rect.centery
    
    # Bricks (3x2 grid)
    brick_colors = [(231, 76, 60), (241, 196, 15), (46, 204, 113)]
    brick_w, brick_h = 18, 8
    start_x = cx - 30
    start_y = cy - 20
    
    for row in range(2):
        for col in range(3):
            bx = start_x + col * (brick_w + 3)
            by = start_y + row * (brick_h + 3)
            pygame.draw.rect(surface, brick_colors[col], (bx, by, brick_w, brick_h), border_radius=2)
    
    # Ball
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy + 15), 5)
    
    # Paddle
    pygame.draw.rect(surface, color, (cx - 20, cy + 30, 40, 8), border_radius=4)


def draw_space_invaders_icon(surface: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int]) -> None:
    """Draw a Space Invaders-style alien icon."""
    cx, cy = rect.centerx, rect.centery - 5
    
    # Classic alien shape (pixel art style)
    alien_color = color
    pixel_size = 4
    
    # Alien pattern (11x8 grid)
    pattern = [
        "  X     X  ",
        "   X   X   ",
        "  XXXXXXX  ",
        " XX XXX XX ",
        "XXXXXXXXXXX",
        "X XXXXXXX X",
        "X X     X X",
        "   XX XX   ",
    ]
    
    start_x = cx - len(pattern[0]) * pixel_size // 2
    start_y = cy - len(pattern) * pixel_size // 2
    
    for row_idx, row in enumerate(pattern):
        for col_idx, char in enumerate(row):
            if char == 'X':
                px = start_x + col_idx * pixel_size
                py = start_y + row_idx * pixel_size
                pygame.draw.rect(surface, alien_color, (px, py, pixel_size, pixel_size))
    
    # Player ship at bottom
    ship_y = cy + 30
    pygame.draw.polygon(surface, (100, 200, 255), [
        (cx, ship_y - 8),
        (cx - 12, ship_y + 5),
        (cx + 12, ship_y + 5)
    ])


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
        self.hover_progress = 0.0
        self.selected = False
        self.pulse_time = 0.0
        
        # Colors from game info
        self.base_color = info.get('color', (100, 181, 246))
        
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def update(self, dt: float) -> None:
        """Update animations."""
        target = 1.0 if self.hover or self.selected else 0.0
        self.hover_progress += (target - self.hover_progress) * min(1.0, dt * 12)
        self.pulse_time += dt
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the game card."""
        # Calculate animated values
        scale = 1.0 + self.hover_progress * 0.03
        pulse = math.sin(self.pulse_time * 3) * 0.5 + 0.5 if self.selected else 0
        
        # Animated rect
        cx, cy = self.x + self.width // 2, self.y + self.height // 2
        w = int(self.width * scale)
        h = int(self.height * scale)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        
        # Glow effect
        if self.hover_progress > 0.1:
            glow_alpha = int(self.hover_progress * 60 + pulse * 20)
            for i in range(3):
                glow_rect = rect.inflate(15 + i * 8, 15 + i * 8)
                glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                glow_color = (*self.base_color, max(0, glow_alpha - i * 20))
                pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=16 + i * 4)
                screen.blit(glow_surface, glow_rect)
        
        # Card background with gradient
        card_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Gradient background
        for y_offset in range(h):
            t = y_offset / h
            r = int(25 + t * 15 + self.hover_progress * 10)
            g = int(30 + t * 15 + self.hover_progress * 10)
            b = int(45 + t * 20 + self.hover_progress * 15)
            pygame.draw.line(card_surface, (r, g, b), (0, y_offset), (w, y_offset))
        
        # Apply rounded corners by masking
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=14)
        card_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        
        screen.blit(card_surface, rect)
        
        # Border with animated color
        border_intensity = 0.3 + self.hover_progress * 0.7
        border_color = tuple(int(c * border_intensity + 50 * (1 - border_intensity)) for c in self.base_color)
        border_width = 2 if self.hover_progress > 0.5 else 1
        pygame.draw.rect(screen, border_color, rect, border_width, border_radius=14)
        
        # Selection indicator - top accent line
        if self.hover_progress > 0.1:
            accent_width = int(w * 0.6 * self.hover_progress)
            accent_rect = pygame.Rect(rect.centerx - accent_width // 2, rect.top, accent_width, 3)
            pygame.draw.rect(screen, self.base_color, accent_rect, border_radius=2)
        
        # Draw custom icon
        icon_rect = pygame.Rect(rect.left, rect.top + 15, rect.width, 70)
        if self.game_id == 'breakout':
            draw_breakout_icon(screen, icon_rect, self.base_color)
        elif self.game_id == 'space_invaders':
            draw_space_invaders_icon(screen, icon_rect, self.base_color)
        else:
            # Generic game icon
            pygame.draw.circle(screen, self.base_color, icon_rect.center, 20, 3)
        
        # Game name
        name = self.info.get('name', self.game_id.replace('_', ' ').title())
        name_font = pygame.font.Font(None, 34)
        name_color = (255, 255, 255) if self.hover_progress > 0.5 else (220, 225, 235)
        name_text = name_font.render(name, True, name_color)
        name_rect = name_text.get_rect(centerx=rect.centerx, top=rect.top + 95)
        screen.blit(name_text, name_rect)
        
        # Description
        desc = self.info.get('description', '')
        desc_font = pygame.font.Font(None, 20)
        desc_text = desc_font.render(desc, True, (130, 140, 160))
        desc_rect = desc_text.get_rect(centerx=rect.centerx, top=name_rect.bottom + 8)
        screen.blit(desc_text, desc_rect)
        
        # Difficulty badge with color coding
        difficulty = self.info.get('difficulty', 'Unknown')
        diff_font = pygame.font.Font(None, 18)
        
        # Color based on difficulty
        if 'easy' in difficulty.lower():
            badge_color = (46, 204, 113)
        elif 'hard' in difficulty.lower():
            badge_color = (231, 76, 60)
        else:
            badge_color = (52, 152, 219)
        
        diff_text = diff_font.render(difficulty, True, badge_color)
        diff_rect = diff_text.get_rect(centerx=rect.centerx, bottom=rect.bottom - 15)
        
        # Badge background
        badge_rect = diff_rect.inflate(20, 8)
        badge_surface = pygame.Surface(badge_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(badge_surface, (*badge_color, 40), badge_surface.get_rect(), border_radius=10)
        pygame.draw.rect(badge_surface, (*badge_color, 100), badge_surface.get_rect(), 1, border_radius=10)
        screen.blit(badge_surface, badge_rect)
        screen.blit(diff_text, diff_rect)


class ActionButton:
    """A styled action button for the menu."""
    
    def __init__(self, text: str, x: int, y: int, key_hint: str = ""):
        self.text = text
        self.key_hint = key_hint
        self.x = x
        self.y = y
        self.hover = False
        
        # Calculate size based on text
        font = pygame.font.Font(None, 22)
        text_surface = font.render(f"{key_hint} {text}" if key_hint else text, True, (255, 255, 255))
        self.width = text_surface.get_width() + 30
        self.height = 36
        
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)
    
    def draw(self, screen: pygame.Surface) -> None:
        rect = self.rect
        
        # Background
        bg_color = (50, 55, 75) if not self.hover else (60, 70, 95)
        pygame.draw.rect(screen, bg_color, rect, border_radius=8)
        
        # Border
        border_color = (80, 90, 120) if not self.hover else (100, 120, 160)
        pygame.draw.rect(screen, border_color, rect, 1, border_radius=8)
        
        # Key hint (styled differently)
        if self.key_hint:
            key_font = pygame.font.Font(None, 18)
            key_text = key_font.render(self.key_hint, True, (100, 180, 255))
            key_rect = key_text.get_rect(left=rect.left + 10, centery=rect.centery)
            
            # Key background
            key_bg = key_rect.inflate(8, 4)
            pygame.draw.rect(screen, (30, 35, 50), key_bg, border_radius=4)
            pygame.draw.rect(screen, (60, 70, 100), key_bg, 1, border_radius=4)
            screen.blit(key_text, key_rect)
            
            # Text
            text_font = pygame.font.Font(None, 22)
            text_surface = text_font.render(self.text, True, (200, 205, 220))
            text_rect = text_surface.get_rect(left=key_bg.right + 8, centery=rect.centery)
            screen.blit(text_surface, text_rect)
        else:
            text_font = pygame.font.Font(None, 22)
            text_surface = text_font.render(self.text, True, (200, 205, 220))
            text_rect = text_surface.get_rect(center=rect.center)
            screen.blit(text_surface, text_rect)


class GameMenu:
    """
    Visual game selection menu.
    
    Displays available games as cards and allows selection
    via mouse or keyboard.
    """
    
    def __init__(self, screen_width: int = 800, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Background colors
        self.bg_color_top = (8, 10, 18)
        self.bg_color_bottom = (18, 22, 35)
        
        # Build game cards
        self.cards: List[GameCard] = []
        self.selected_index = 0
        self._build_cards()
        
        # Action buttons
        button_y = screen_height - 50
        self.buttons = [
            ActionButton("Navigate", screen_width // 2 - 180, button_y, "< >"),
            ActionButton("Start", screen_width // 2, button_y, "Enter"),
            ActionButton("Quit", screen_width // 2 + 150, button_y, "Esc"),
        ]
        
        # Animation
        self.time = 0.0
        
        # Animated background particles
        self.particles = []
        for i in range(60):
            self.particles.append({
                'x': screen_width * (i * 0.618 % 1),
                'y': screen_height * (i * 0.382 % 1),
                'size': 1 + (i % 3),
                'speed': 0.2 + (i % 5) * 0.1,
                'phase': i * 0.5
            })
        
        # Diagonal lines for background effect
        self.diag_lines = []
        for i in range(8):
            self.diag_lines.append({
                'offset': i * 150,
                'speed': 0.3 + i * 0.05,
                'alpha': 15 + i * 3
            })
    
    def _build_cards(self) -> None:
        """Build game cards from registry."""
        from . import list_games, get_game_info
        
        games = list_games()
        num_games = len(games)
        
        if num_games == 0:
            return
        
        # Card dimensions
        card_width = 220
        card_height = 220
        card_spacing = 40
        
        # Calculate total width
        total_width = num_games * card_width + (num_games - 1) * card_spacing
        start_x = (self.screen_width - total_width) // 2
        y = (self.screen_height - card_height) // 2 + 20
        
        for i, game_id in enumerate(games):
            info = get_game_info(game_id) or {}
            x = start_x + i * (card_width + card_spacing)
            
            card = GameCard(game_id, x, y, card_width, card_height, info)
            self.cards.append(card)
        
        # Select first card
        if self.cards:
            self.cards[0].selected = True
    
    def run(self, screen: pygame.Surface, clock: pygame.time.Clock) -> Optional[str]:
        """Run the menu and return selected game."""
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
            
            # Update particles
            for p in self.particles:
                p['y'] -= p['speed']
                if p['y'] < -10:
                    p['y'] = self.screen_height + 10
            
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
        
        # Button hover
        for button in self.buttons:
            button.hover = button.rect.collidepoint(pos)
    
    def _handle_click(self, pos: Tuple[int, int]) -> Optional[str]:
        """Handle mouse click."""
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
        
        # Animated diagonal lines
        for line in self.diag_lines:
            offset = (self.time * 50 * line['speed'] + line['offset']) % (self.screen_width + self.screen_height)
            start = (offset, 0)
            end = (offset - self.screen_height, self.screen_height)
            
            line_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            pygame.draw.line(line_surface, (255, 255, 255, line['alpha']), start, end, 1)
            screen.blit(line_surface, (0, 0))
        
        # Particles (stars)
        for p in self.particles:
            alpha = int(80 + 40 * math.sin(self.time * 2 + p['phase']))
            color = (alpha + 50, alpha + 60, alpha + 80)
            pygame.draw.circle(screen, color, (int(p['x']), int(p['y'])), p['size'])
        
        # Title with glow effect
        title_font = pygame.font.Font(None, 72)
        title_text = "Neural Network AI"
        
        # Glow layers
        for i in range(3):
            glow_surface = title_font.render(title_text, True, (0, 150, 255, 30 - i * 10))
            glow_rect = glow_surface.get_rect(centerx=self.screen_width // 2 + (i - 1), top=55 + (i - 1))
            screen.blit(glow_surface, glow_rect)
        
        # Main title
        title_surface = title_font.render(title_text, True, (255, 255, 255))
        title_rect = title_surface.get_rect(centerx=self.screen_width // 2, top=55)
        screen.blit(title_surface, title_rect)
        
        # Decorative line under title
        line_width = 200
        line_y = title_rect.bottom + 15
        pygame.draw.line(screen, (50, 60, 90), 
                        (self.screen_width // 2 - line_width, line_y),
                        (self.screen_width // 2 + line_width, line_y), 1)
        # Accent in center
        accent_width = 60
        pygame.draw.line(screen, (0, 180, 255),
                        (self.screen_width // 2 - accent_width, line_y),
                        (self.screen_width // 2 + accent_width, line_y), 2)
        
        # Subtitle
        sub_font = pygame.font.Font(None, 30)
        sub_text = sub_font.render("Select a game to train", True, (120, 130, 160))
        sub_rect = sub_text.get_rect(centerx=self.screen_width // 2, top=line_y + 20)
        screen.blit(sub_text, sub_rect)
        
        # Draw cards
        for card in self.cards:
            card.draw(screen)
        
        # Draw action buttons
        for button in self.buttons:
            button.draw(screen)


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
