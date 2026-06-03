"""
A simple working agent for testing
"""
from environment import Move
import random
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent

class PacmanAgent(BasePacmanAgent):
    def __init__(self, pacman_speed: int = 1):
        super().__init__()
        self.role = 'pacman'
        self.pacman_speed = pacman_speed
    
    def step(self, observation, my_position, enemy_position, current_step):
        """Move randomly."""
        return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])

class GhostAgent(BaseGhostAgent):
    def __init__(self):
        super().__init__()
        self.role = 'ghost'
    
    def step(self, observation, my_position, enemy_position, current_step):
        """Move randomly."""
        return random.choice([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT])
