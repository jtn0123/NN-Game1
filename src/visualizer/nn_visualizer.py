"""
Neural Network Visualizer
=========================

Real-time visualization of the neural network structure and activations.

Features:
    - Display network architecture (layers, neurons, connections)
    - Show live activations as colors (red = negative, green = positive)
    - Animate connection weights
    - Highlight selected action
    - Display Q-values for each action

This creates a beautiful, informative visualization that helps you
understand what the network is "thinking" in real-time.
"""

import pygame
import numpy as np
from typing import Optional, List, Tuple, Dict
import math

import sys
sys.path.append('../..')
from config import Config


class NeuralNetVisualizer:
    """
    Visualizes the neural network in real-time.
    
    Shows:
        - Network layers and neurons
        - Activations (color-coded by value)
        - Connection weights (line thickness/color)
        - Q-values and selected action
    
    The visualization updates every frame to show the current
    state of the network as it processes game states.
    
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
        
        # Colors
        self.bg_color = (20, 20, 30)
        self.text_color = (200, 200, 200)
        self.inactive_color = self.config.VIS_COLOR_INACTIVE
        self.active_positive = (0, 255, 128)  # Green for positive
        self.active_negative = (255, 80, 80)   # Red for negative
        
        # Font
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        
        # Action labels
        self.action_labels = ["← LEFT", "• STAY", "RIGHT →"]
        
        # Animation state
        self.pulse_phase = 0
    
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
        # Draw background panel
        self._draw_background(screen)
        
        # Get network info
        layer_info = agent.policy_net.get_layer_info()
        activations = agent.policy_net.get_activations()
        weights = agent.policy_net.get_weights()
        q_values = agent.get_q_values(state)
        
        # Calculate layer positions
        layer_positions = self._calculate_layer_positions(layer_info)
        
        # Draw connections first (so they're behind neurons)
        self._draw_connections(screen, layer_positions, weights, activations)
        
        # Draw neurons
        self._draw_neurons(screen, layer_positions, layer_info, activations, state)
        
        # Draw Q-values and action selection
        self._draw_q_values(screen, q_values, selected_action)
        
        # Draw title
        self._draw_title(screen)
        
        # Update animation
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * math.pi)
    
    def _draw_background(self, screen: pygame.Surface) -> None:
        """Draw the visualization background panel."""
        # Main background
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.bg_color, panel_rect)
        
        # Border with glow effect
        border_color = (60, 60, 80)
        pygame.draw.rect(screen, border_color, panel_rect, 2)
        
        # Inner glow
        inner_rect = panel_rect.inflate(-4, -4)
        pygame.draw.rect(screen, (30, 30, 45), inner_rect, 1)
    
    def _draw_title(self, screen: pygame.Surface) -> None:
        """Draw the visualization title."""
        title = "Neural Network"
        text = self.font_large.render(title, True, (100, 200, 255))
        text_rect = text.get_rect(centerx=self.x + self.width // 2, top=self.y + 10)
        screen.blit(text, text_rect)
        
        # Subtitle
        subtitle = "Live Activations"
        sub_text = self.font_small.render(subtitle, True, (150, 150, 150))
        sub_rect = sub_text.get_rect(centerx=self.x + self.width // 2, top=self.y + 40)
        screen.blit(sub_text, sub_rect)
    
    def _calculate_layer_positions(self, layer_info: List[Dict]) -> List[Dict]:
        """
        Calculate the position of each layer and its neurons.
        
        Returns:
            List of dicts with 'x', 'neurons', and 'positions' for each layer
        """
        num_layers = len(layer_info)
        layer_spacing = (self.width - 60) / max(num_layers - 1, 1)
        
        positions = []
        
        for i, info in enumerate(layer_info):
            layer_x = self.x + 30 + i * layer_spacing
            num_neurons = min(info['neurons'], self.max_neurons)
            
            # Calculate vertical positions
            available_height = self.height - 160
            neuron_spacing = min(
                self.config.VIS_NEURON_SPACING,
                available_height / max(num_neurons, 1)
            )
            
            total_height = num_neurons * neuron_spacing
            start_y = self.y + 80 + (available_height - total_height) / 2
            
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
        """Draw connections between layers with weight-based styling."""
        for i in range(len(layer_positions) - 1):
            from_layer = layer_positions[i]
            to_layer = layer_positions[i + 1]
            
            if i < len(weights):
                weight_matrix = weights[i]
                max_weight = np.abs(weight_matrix).max() + 1e-6
                
                # Sample connections if there are too many
                from_indices = list(range(len(from_layer['positions'])))
                to_indices = list(range(len(to_layer['positions'])))
                
                # Only draw a subset of connections for clarity
                max_connections = 50
                if len(from_indices) * len(to_indices) > max_connections:
                    from_sample = np.random.choice(
                        from_indices, 
                        size=min(5, len(from_indices)), 
                        replace=False
                    )
                    to_sample = np.random.choice(
                        to_indices, 
                        size=min(10, len(to_indices)), 
                        replace=False
                    )
                else:
                    from_sample = from_indices
                    to_sample = to_indices
                
                for fi in from_sample:
                    for ti in to_sample:
                        if fi < from_layer['actual_neurons'] and ti < weight_matrix.shape[0]:
                            # Adjust indices for weight matrix
                            weight_idx = min(fi, weight_matrix.shape[1] - 1)
                            weight = weight_matrix[ti, weight_idx]
                            
                            # Normalize weight for visualization
                            norm_weight = weight / max_weight
                            
                            # Color based on weight sign
                            if norm_weight > 0:
                                color = self._interpolate_color(
                                    (50, 50, 50), 
                                    self.active_positive, 
                                    abs(norm_weight)
                                )
                            else:
                                color = self._interpolate_color(
                                    (50, 50, 50), 
                                    self.active_negative, 
                                    abs(norm_weight)
                                )
                            
                            # Line thickness based on weight magnitude
                            thickness = max(1, int(abs(norm_weight) * 3))
                            
                            # Draw connection
                            from_pos = from_layer['positions'][fi]
                            to_pos = to_layer['positions'][ti]
                            
                            pygame.draw.line(
                                screen, color,
                                (int(from_pos[0]), int(from_pos[1])),
                                (int(to_pos[0]), int(to_pos[1])),
                                thickness
                            )
    
    def _draw_neurons(
        self,
        screen: pygame.Surface,
        layer_positions: List[Dict],
        layer_info: List[Dict],
        activations: Dict[str, np.ndarray],
        state: np.ndarray
    ) -> None:
        """Draw neurons with activation-based coloring."""
        for i, (layer_pos, info) in enumerate(zip(layer_positions, layer_info)):
            # Get activations for this layer
            if info['type'] == 'input':
                # Input layer uses state values
                layer_acts = state[:layer_pos['neurons']]
            elif f'layer_{i-1}' in activations:
                acts = activations[f'layer_{i-1}']
                if len(acts.shape) > 1:
                    acts = acts[0]  # First batch item
                layer_acts = acts[:layer_pos['neurons']]
            else:
                layer_acts = np.zeros(layer_pos['neurons'])
            
            # Normalize activations
            if len(layer_acts) > 0 and np.abs(layer_acts).max() > 0:
                norm_acts = layer_acts / (np.abs(layer_acts).max() + 1e-6)
            else:
                norm_acts = layer_acts
            
            for j, pos in enumerate(layer_pos['positions']):
                # Get activation value
                act_val = norm_acts[j] if j < len(norm_acts) else 0
                
                # Calculate color based on activation
                if act_val > 0:
                    color = self._interpolate_color(
                        self.inactive_color, 
                        self.active_positive, 
                        min(abs(act_val), 1.0)
                    )
                else:
                    color = self._interpolate_color(
                        self.inactive_color, 
                        self.active_negative, 
                        min(abs(act_val), 1.0)
                    )
                
                # Pulse effect for highly active neurons
                radius = self.neuron_radius
                if abs(act_val) > 0.8:
                    pulse = math.sin(self.pulse_phase) * 0.2 + 1
                    radius = int(radius * pulse)
                
                # Draw outer glow for active neurons
                if abs(act_val) > 0.5:
                    glow_radius = radius + 3
                    glow_color = tuple(min(255, c + 30) for c in color)
                    pygame.draw.circle(
                        screen, glow_color,
                        (int(pos[0]), int(pos[1])),
                        glow_radius
                    )
                
                # Draw neuron
                pygame.draw.circle(
                    screen, color,
                    (int(pos[0]), int(pos[1])),
                    radius
                )
                
                # Draw border
                pygame.draw.circle(
                    screen, (100, 100, 120),
                    (int(pos[0]), int(pos[1])),
                    radius, 1
                )
            
            # Draw ellipsis if there are more neurons than displayed
            if layer_pos['neurons'] < info['neurons']:
                ellipsis_y = layer_pos['positions'][-1][1] + 25
                text = self.font_small.render(
                    f"... +{info['neurons'] - layer_pos['neurons']} more",
                    True, (120, 120, 120)
                )
                text_rect = text.get_rect(
                    centerx=int(layer_pos['x']), 
                    centery=int(ellipsis_y)
                )
                screen.blit(text, text_rect)
    
    def _draw_q_values(
        self,
        screen: pygame.Surface,
        q_values: np.ndarray,
        selected_action: Optional[int] = None
    ) -> None:
        """Draw Q-values and highlight selected action."""
        # Q-value display area
        qv_y = self.y + self.height - 80
        qv_height = 70
        
        # Background for Q-value area
        qv_rect = pygame.Rect(self.x + 10, qv_y, self.width - 20, qv_height)
        pygame.draw.rect(screen, (25, 25, 40), qv_rect, border_radius=5)
        pygame.draw.rect(screen, (60, 60, 80), qv_rect, 1, border_radius=5)
        
        # Title
        title = self.font_small.render("Q-Values (Action Quality)", True, (150, 150, 150))
        screen.blit(title, (self.x + 15, qv_y + 5))
        
        # Normalize Q-values for visualization
        q_min, q_max = q_values.min(), q_values.max()
        q_range = q_max - q_min + 1e-6
        
        # Draw bars for each action
        bar_width = (self.width - 60) / len(q_values)
        bar_max_height = 35
        
        for i, (q_val, label) in enumerate(zip(q_values, self.action_labels)):
            bar_x = self.x + 20 + i * bar_width
            
            # Normalize height
            norm_q = (q_val - q_min) / q_range
            bar_height = max(5, int(norm_q * bar_max_height))
            
            # Color based on whether this is the selected action
            if i == selected_action or (selected_action is None and i == np.argmax(q_values)):
                color = (0, 200, 100)  # Bright green for selected
                border_color = (100, 255, 150)
            else:
                color = (80, 80, 100)  # Gray for unselected
                border_color = (100, 100, 120)
            
            # Draw bar
            bar_rect = pygame.Rect(
                int(bar_x), 
                int(qv_y + 50 - bar_height),
                int(bar_width - 10), 
                bar_height
            )
            pygame.draw.rect(screen, color, bar_rect, border_radius=3)
            pygame.draw.rect(screen, border_color, bar_rect, 1, border_radius=3)
            
            # Draw label
            label_text = self.font_small.render(label, True, (180, 180, 180))
            label_rect = label_text.get_rect(
                centerx=int(bar_x + bar_width / 2 - 5),
                top=int(qv_y + 55)
            )
            screen.blit(label_text, label_rect)
    
    def _interpolate_color(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        t: float
    ) -> Tuple[int, int, int]:
        """Interpolate between two colors."""
        t = max(0, min(1, t))
        return tuple(
            int(c1 + (c2 - c1) * t)
            for c1, c2 in zip(color1, color2)
        )


# Testing
if __name__ == "__main__":
    print("NeuralNetVisualizer - import and use with pygame")
    print("See main.py for usage example")

