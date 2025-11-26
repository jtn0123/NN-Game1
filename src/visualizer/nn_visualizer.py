"""
Neural Network Visualizer - Enhanced
====================================

Real-time visualization of the neural network structure and activations.

Features:
    - Display network architecture (layers, neurons, connections)
    - Show live activations with smooth interpolation
    - Animated data flow pulses traveling through connections
    - Gradient coloring (blue-white-red diverging palette)
    - Layer labels with neuron counts
    - Highlight selected action with animated emphasis
    - Display Q-values for each action

This creates a beautiful, informative visualization that helps you
understand what the network is "thinking" in real-time.
"""

import pygame
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from collections import deque
import math
import time

import sys
sys.path.append('../..')
from config import Config


class DataFlowPulse:
    """Represents an animated pulse traveling through a connection."""
    
    def __init__(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                 color: Tuple[int, int, int], speed: float = 0.05):
        self.start = start_pos
        self.end = end_pos
        self.color = color
        self.progress = 0.0
        self.speed = speed
        self.alive = True
    
    def update(self) -> bool:
        self.progress += self.speed
        if self.progress >= 1.0:
            self.alive = False
        return self.alive
    
    @property
    def position(self) -> Tuple[int, int]:
        x = self.start[0] + (self.end[0] - self.start[0]) * self.progress
        y = self.start[1] + (self.end[1] - self.start[1]) * self.progress
        return (int(x), int(y))


class NeuralNetVisualizer:
    """
    Enhanced neural network visualizer with smooth animations.
    
    Shows:
        - Network layers and neurons with smooth activation transitions
        - Animated data flow pulses through connections
        - Diverging color palette for weights (blue-white-red)
        - Layer labels with neuron counts
        - Q-values and action selection with emphasis
    
    Example:
        >>> visualizer = NeuralNetVisualizer(config, x=500, y=50, width=280, height=500)
        >>> visualizer.render(screen, agent, current_state)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        x: int = 0,
        y: int = 0,
        width: int = 300,
        height: int = 500
    ):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration object
            x: X position of visualization area
            y: Y position of visualization area
            width: Width of visualization area
            height: Height of visualization area
        """
        self.config = config or Config()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Visualization parameters
        self.neuron_radius = self.config.VIS_NEURON_RADIUS
        self.max_neurons = self.config.VIS_MAX_NEURONS_DISPLAY
        
        # Enhanced color palette
        self.bg_color = (12, 12, 24)
        self.panel_color = (18, 18, 32)
        self.text_color = (200, 200, 220)
        self.inactive_color = (40, 40, 55)
        
        # Diverging color palette for activations
        self.color_negative = (66, 135, 245)    # Blue for negative
        self.color_neutral = (200, 200, 210)    # White-ish for zero
        self.color_positive = (245, 66, 108)    # Red/pink for positive
        
        # Weight colors (blue-white-red diverging)
        self.weight_negative = (41, 98, 255)    # Blue
        self.weight_neutral = (100, 100, 120)   # Gray
        self.weight_positive = (255, 98, 41)    # Orange/red
        
        # Font initialization
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        self.font_title = pygame.font.Font(None, 36)
        
        # Action labels
        self.action_labels = ["LEFT", "STAY", "RIGHT"]
        self.action_icons = ["◀", "●", "▶"]
        
        # Animation state
        self.pulse_phase = 0.0
        self.time_offset = time.time()
        
        # Smooth activation interpolation
        self.prev_activations: Dict[str, np.ndarray] = {}
        self.interpolation_speed = 0.3
        
        # Data flow pulses
        self.pulses: List[DataFlowPulse] = []
        self.pulse_spawn_timer = 0
        self.pulse_spawn_interval = 5  # Frames between pulse spawns
        
        # Activation history for mini sparklines
        self.activation_history: Dict[str, deque] = {}
        self.history_length = 30
        
        # Cached layer positions
        self._cached_positions: Optional[List[Dict[str, Any]]] = None
        self._cached_layer_info: Optional[str] = None
    
    def render(
        self,
        screen: pygame.Surface,
        agent,
        state: np.ndarray,
        selected_action: Optional[int] = None
    ) -> None:
        """
        Render the neural network visualization.
        
        Args:
            screen: Pygame surface to draw on
            agent: The DQN agent
            state: Current game state
            selected_action: Currently selected action (if any)
        """
        # Get network info
        layer_info = agent.policy_net.get_layer_info()
        activations = agent.policy_net.get_activations()
        weights = agent.policy_net.get_weights()
        q_values = agent.get_q_values(state)
        
        # Calculate layer positions (cache for performance)
        if self._cached_layer_info != str(layer_info):
            self._cached_positions = self._calculate_layer_positions(layer_info)
            self._cached_layer_info = str(layer_info)
        
        # Ensure layer_positions is not None
        layer_positions = self._cached_positions
        if layer_positions is None:
            return
        
        # Interpolate activations for smooth animation
        smoothed_activations = self._smooth_activations(activations)
        
        # Draw background panel
        self._draw_background(screen)
        
        # Draw connections with animated pulses
        self._draw_connections(screen, layer_positions, weights, smoothed_activations)
        
        # Update and draw data flow pulses
        self._update_pulses(layer_positions, smoothed_activations)
        self._draw_pulses(screen)
        
        # Draw neurons with smooth activation coloring
        self._draw_neurons(screen, layer_positions, layer_info, smoothed_activations, state)
        
        # Draw layer labels
        self._draw_layer_labels(screen, layer_positions, layer_info)
        
        # Draw Q-values and action selection
        self._draw_q_values(screen, q_values, selected_action)
        
        # Draw title and info
        self._draw_title(screen, agent)
        
        # Update animation phase
        self.pulse_phase = (self.pulse_phase + 0.08) % (2 * math.pi)
    
    def _smooth_activations(self, activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Interpolate activations for smoother animation."""
        smoothed = {}
        
        for key, new_act in activations.items():
            if len(new_act.shape) > 1:
                new_act = new_act[0]
            
            if key in self.prev_activations:
                # Interpolate between previous and current
                prev = self.prev_activations[key]
                if prev.shape == new_act.shape:
                    smoothed[key] = prev + (new_act - prev) * self.interpolation_speed
                else:
                    smoothed[key] = new_act
            else:
                smoothed[key] = new_act
            
            self.prev_activations[key] = smoothed[key].copy()
            
            # Update history for sparklines
            if key not in self.activation_history:
                self.activation_history[key] = deque(maxlen=self.history_length)
            self.activation_history[key].append(np.mean(np.abs(smoothed[key])))
        
        return smoothed
    
    def _draw_background(self, screen: pygame.Surface) -> None:
        """Draw enhanced visualization background panel."""
        # Main background with gradient effect
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
        # Draw gradient background
        for i in range(self.height):
            progress = i / self.height
            r = int(self.bg_color[0] + (self.panel_color[0] - self.bg_color[0]) * progress)
            g = int(self.bg_color[1] + (self.panel_color[1] - self.bg_color[1]) * progress)
            b = int(self.bg_color[2] + (self.panel_color[2] - self.bg_color[2]) * progress)
            pygame.draw.line(screen, (r, g, b), 
                           (self.x, self.y + i), (self.x + self.width, self.y + i))
        
        # Animated border glow
        glow_intensity = int(20 + 10 * math.sin(self.pulse_phase))
        border_color = (40 + glow_intensity, 60 + glow_intensity, 100 + glow_intensity)
        pygame.draw.rect(screen, border_color, panel_rect, 2, border_radius=8)
        
        # Inner border
        inner_rect = panel_rect.inflate(-4, -4)
        pygame.draw.rect(screen, (30, 35, 50), inner_rect, 1, border_radius=6)
    
    def _draw_title(self, screen: pygame.Surface, agent) -> None:
        """Draw the visualization title with training info."""
        # Main title
        title = "Neural Network"
        text = self.font_title.render(title, True, (100, 180, 255))
        text_rect = text.get_rect(centerx=self.x + self.width // 2, top=self.y + 8)
        screen.blit(text, text_rect)
        
        # Animated subtitle
        pulse = 0.7 + 0.3 * math.sin(self.pulse_phase * 0.5)
        sub_color = (int(100 * pulse), int(200 * pulse), int(150 * pulse))
        subtitle = "● LIVE"
        sub_text = self.font_small.render(subtitle, True, sub_color)
        sub_rect = sub_text.get_rect(centerx=self.x + self.width // 2, top=self.y + 36)
        screen.blit(sub_text, sub_rect)
        
        # Training step counter
        step_text = self.font_small.render(f"Step: {agent.steps:,}", True, (120, 120, 140))
        screen.blit(step_text, (self.x + 10, self.y + 8))
    
    def _calculate_layer_positions(self, layer_info: List[Dict]) -> List[Dict]:
        """Calculate the position of each layer and its neurons."""
        num_layers = len(layer_info)
        layer_spacing = (self.width - 80) / max(num_layers - 1, 1)
        
        positions = []
        
        for i, info in enumerate(layer_info):
            layer_x = self.x + 40 + i * layer_spacing
            num_neurons = min(info['neurons'], self.max_neurons)
            
            # Calculate vertical positions
            available_height = self.height - 200
            neuron_spacing = min(
                self.config.VIS_NEURON_SPACING,
                available_height / max(num_neurons, 1)
            )
            
            total_height = num_neurons * neuron_spacing
            start_y = self.y + 70 + (available_height - total_height) / 2
            
            neuron_positions = []
            for j in range(num_neurons):
                ny = start_y + j * neuron_spacing + neuron_spacing / 2
                neuron_positions.append((layer_x, ny))
            
            positions.append({
                'x': layer_x,
                'neurons': num_neurons,
                'actual_neurons': info['neurons'],
                'positions': neuron_positions,
                'type': info['type'],
                'name': info['name']
            })
        
        return positions
    
    def _draw_connections(
        self,
        screen: pygame.Surface,
        layer_positions: List[Dict],
        weights: List[np.ndarray],
        activations: Dict[str, np.ndarray]
    ) -> None:
        """Draw connections with enhanced weight visualization."""
        for i in range(len(layer_positions) - 1):
            from_layer = layer_positions[i]
            to_layer = layer_positions[i + 1]
            
            if i < len(weights):
                weight_matrix = weights[i]
                max_weight = np.abs(weight_matrix).max() + 1e-6
                
                # Sample connections
                from_indices = list(range(len(from_layer['positions'])))
                to_indices = list(range(len(to_layer['positions'])))
                
                max_connections = 60
                from_sample: List[int]
                to_sample: List[int]
                if len(from_indices) * len(to_indices) > max_connections:
                    np.random.seed(42)  # Consistent sampling
                    from_sample = list(np.random.choice(from_indices, size=min(6, len(from_indices)), replace=False))
                    to_sample = list(np.random.choice(to_indices, size=min(10, len(to_indices)), replace=False))
                else:
                    from_sample = from_indices
                    to_sample = to_indices
                
                for fi in from_sample:
                    for ti in to_sample:
                        if fi < from_layer['actual_neurons'] and ti < weight_matrix.shape[0]:
                            weight_idx = min(fi, weight_matrix.shape[1] - 1)
                            weight = weight_matrix[ti, weight_idx]
                            norm_weight = weight / max_weight
                            
                            # Diverging color: blue (neg) -> gray -> orange (pos)
                            if norm_weight > 0:
                                color = self._interpolate_color(
                                    self.weight_neutral,
                                    self.weight_positive,
                                    abs(norm_weight)
                                )
                            else:
                                color = self._interpolate_color(
                                    self.weight_neutral,
                                    self.weight_negative,
                                    abs(norm_weight)
                                )
                            
                            # Line thickness
                            thickness = max(1, int(abs(norm_weight) * 2.5))
                            
                            from_pos = from_layer['positions'][fi]
                            to_pos = to_layer['positions'][ti]
                            
                            # Draw with slight transparency effect
                            pygame.draw.line(
                                screen, color,
                                (int(from_pos[0]), int(from_pos[1])),
                                (int(to_pos[0]), int(to_pos[1])),
                                thickness
                            )
    
    def _update_pulses(self, layer_positions: List[Dict], activations: Dict[str, np.ndarray]) -> None:
        """Update and spawn data flow pulses."""
        # Update existing pulses
        self.pulses = [p for p in self.pulses if p.update()]
        
        # Spawn new pulses periodically
        self.pulse_spawn_timer += 1
        if self.pulse_spawn_timer >= self.pulse_spawn_interval and len(layer_positions) > 1:
            self.pulse_spawn_timer = 0
            
            # Spawn pulses from active neurons
            for i in range(len(layer_positions) - 1):
                from_layer = layer_positions[i]
                to_layer = layer_positions[i + 1]
                
                # Get activation level to determine pulse color
                layer_key = f'layer_{i}'
                if layer_key in activations:
                    act_level = np.mean(np.abs(activations[layer_key]))
                else:
                    act_level = 0.5
                
                # Spawn a few pulses
                if len(from_layer['positions']) > 0 and len(to_layer['positions']) > 0:
                    for _ in range(2):
                        from_idx = np.random.randint(0, len(from_layer['positions']))
                        to_idx = np.random.randint(0, len(to_layer['positions']))
                        
                        from_pos = from_layer['positions'][from_idx]
                        to_pos = to_layer['positions'][to_idx]
                        
                        # Color based on activation
                        intensity = min(1.0, act_level * 2)
                        color = self._interpolate_color(
                            (60, 80, 120),
                            (100, 255, 180),
                            intensity
                        )
                        
                        pulse = DataFlowPulse(from_pos, to_pos, color, speed=0.08)
                        self.pulses.append(pulse)
    
    def _draw_pulses(self, screen: pygame.Surface) -> None:
        """Draw data flow pulses."""
        for pulse in self.pulses:
            pos = pulse.position
            # Fade based on progress
            alpha = 1.0 - abs(pulse.progress - 0.5) * 2
            size = int(3 + 2 * alpha)
            
            color = tuple(int(c * alpha) for c in pulse.color)
            
            # Draw glow
            pygame.draw.circle(screen, color, pos, size + 2)
            # Draw core
            bright_color = tuple(min(255, int(c * 1.5)) for c in pulse.color)
            pygame.draw.circle(screen, bright_color, pos, size)
    
    def _draw_neurons(
        self,
        screen: pygame.Surface,
        layer_positions: List[Dict],
        layer_info: List[Dict],
        activations: Dict[str, np.ndarray],
        state: np.ndarray
    ) -> None:
        """Draw neurons with enhanced activation visualization."""
        for i, (layer_pos, info) in enumerate(zip(layer_positions, layer_info)):
            # Get activations
            if info['type'] == 'input':
                layer_acts = state[:layer_pos['neurons']]
            elif f'layer_{i-1}' in activations:
                layer_acts = activations[f'layer_{i-1}'][:layer_pos['neurons']]
            else:
                layer_acts = np.zeros(layer_pos['neurons'])
            
            # Normalize
            max_act = np.abs(layer_acts).max() if len(layer_acts) > 0 else 1
            norm_acts = layer_acts / (max_act + 1e-6)
            
            for j, pos in enumerate(layer_pos['positions']):
                act_val = norm_acts[j] if j < len(norm_acts) else 0
                
                # Diverging color based on activation sign
                if act_val > 0:
                    color = self._interpolate_color(
                        self.inactive_color,
                        self.color_positive,
                        min(abs(act_val), 1.0)
                    )
                else:
                    color = self._interpolate_color(
                        self.inactive_color,
                        self.color_negative,
                        min(abs(act_val), 1.0)
                    )
                
                # Pulse effect for highly active neurons
                radius = self.neuron_radius
                if abs(act_val) > 0.7:
                    pulse = 1 + 0.2 * math.sin(self.pulse_phase + j * 0.5)
                    radius = int(radius * pulse)
                
                # Draw outer glow for active neurons
                if abs(act_val) > 0.4:
                    glow_radius = radius + 4
                    glow_alpha = abs(act_val) * 0.6
                    glow_color = tuple(int(c * glow_alpha) for c in color)
                    pygame.draw.circle(screen, glow_color, (int(pos[0]), int(pos[1])), glow_radius)
                
                # Draw neuron body
                pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), radius)
                
                # Draw highlight (3D effect)
                highlight_pos = (int(pos[0] - radius * 0.3), int(pos[1] - radius * 0.3))
                highlight_radius = max(1, radius // 3)
                highlight_color = tuple(min(255, c + 50) for c in color)
                pygame.draw.circle(screen, highlight_color, highlight_pos, highlight_radius)
                
                # Draw border
                pygame.draw.circle(screen, (80, 90, 110), (int(pos[0]), int(pos[1])), radius, 1)
            
            # Draw ellipsis for hidden neurons
            if layer_pos['neurons'] < info['neurons']:
                ellipsis_y = layer_pos['positions'][-1][1] + 20
                text = self.font_small.render(
                    f"+{info['neurons'] - layer_pos['neurons']}",
                    True, (100, 100, 120)
                )
                text_rect = text.get_rect(centerx=int(layer_pos['x']), centery=int(ellipsis_y))
                screen.blit(text, text_rect)
    
    def _draw_layer_labels(self, screen: pygame.Surface, layer_positions: List[Dict], 
                           layer_info: List[Dict]) -> None:
        """Draw layer labels with neuron counts."""
        for layer_pos, info in zip(layer_positions, layer_info):
            # Layer name
            name = info['name']
            if info['type'] == 'input':
                name = "IN"
                color = (100, 180, 255)
            elif info['type'] == 'output':
                name = "OUT"
                color = (255, 150, 100)
            else:
                name = f"H{name.split()[-1]}" if 'Hidden' in name else name
                color = (150, 200, 150)
            
            # Draw label above layer
            label_y = layer_pos['positions'][0][1] - 25 if layer_pos['positions'] else self.y + 60
            text = self.font_small.render(name, True, color)
            text_rect = text.get_rect(centerx=int(layer_pos['x']), bottom=int(label_y))
            screen.blit(text, text_rect)
            
            # Neuron count below
            count_text = self.font_small.render(f"({layer_pos['actual_neurons']})", True, (90, 90, 110))
            count_rect = count_text.get_rect(centerx=int(layer_pos['x']), top=int(label_y) + 2)
            screen.blit(count_text, count_rect)
    
    def _draw_q_values(
        self,
        screen: pygame.Surface,
        q_values: np.ndarray,
        selected_action: Optional[int] = None
    ) -> None:
        """Draw enhanced Q-values visualization."""
        qv_y = self.y + self.height - 90
        qv_height = 80
        
        # Background panel
        qv_rect = pygame.Rect(self.x + 8, qv_y, self.width - 16, qv_height)
        pygame.draw.rect(screen, (20, 22, 35), qv_rect, border_radius=8)
        pygame.draw.rect(screen, (50, 55, 75), qv_rect, 1, border_radius=8)
        
        # Title
        title = self.font_small.render("Q-Values", True, (140, 140, 160))
        screen.blit(title, (self.x + 15, qv_y + 5))
        
        # Normalize Q-values
        q_min, q_max = q_values.min(), q_values.max()
        q_range = q_max - q_min + 1e-6
        
        bar_width = (self.width - 50) / len(q_values)
        bar_max_height = 40
        best_action = np.argmax(q_values)
        
        for i, (q_val, label, icon) in enumerate(zip(q_values, self.action_labels, self.action_icons)):
            bar_x = self.x + 20 + i * bar_width
            
            norm_q = (q_val - q_min) / q_range
            bar_height = max(8, int(norm_q * bar_max_height))
            
            # Determine if this is the selected/best action
            is_selected = (i == selected_action) or (selected_action is None and i == best_action)
            
            if is_selected:
                # Animated selected action
                pulse = 0.8 + 0.2 * math.sin(self.pulse_phase * 2)
                color = (int(50 * pulse), int(220 * pulse), int(120 * pulse))
                border_color = (100, 255, 160)
                # Glow effect
                glow_rect = pygame.Rect(int(bar_x) - 2, int(qv_y + 55 - bar_height) - 2,
                                       int(bar_width - 8) + 4, bar_height + 4)
                pygame.draw.rect(screen, (30, 100, 60), glow_rect, border_radius=4)
            else:
                color = (55, 60, 80)
                border_color = (80, 85, 100)
            
            # Draw bar
            bar_rect = pygame.Rect(int(bar_x), int(qv_y + 55 - bar_height),
                                  int(bar_width - 10), bar_height)
            pygame.draw.rect(screen, color, bar_rect, border_radius=4)
            pygame.draw.rect(screen, border_color, bar_rect, 1, border_radius=4)
            
            # Draw icon and label
            icon_color = (220, 220, 240) if is_selected else (120, 120, 140)
            icon_text = self.font_medium.render(icon, True, icon_color)
            icon_rect = icon_text.get_rect(centerx=int(bar_x + bar_width / 2 - 5), top=int(qv_y + 60))
            screen.blit(icon_text, icon_rect)
            
            # Q-value number
            q_str = f"{q_val:.2f}"
            q_text = self.font_small.render(q_str, True, (100, 100, 120))
            q_rect = q_text.get_rect(centerx=int(bar_x + bar_width / 2 - 5), 
                                     bottom=int(qv_y + 55 - bar_height - 2))
            screen.blit(q_text, q_rect)
    
    def _interpolate_color(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        t: float
    ) -> Tuple[int, int, int]:
        """Interpolate between two colors with easing."""
        t = max(0, min(1, t))
        # Apply ease-out for smoother transitions
        t = 1 - (1 - t) ** 2
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        return (r, g, b)


# Testing
if __name__ == "__main__":
    print("NeuralNetVisualizer - Enhanced version")
    print("See main.py for usage example")
