from dataclasses import dataclass

@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    
    @staticmethod
    def dtype():
        return [('x', 'f8'), ('y', 'f8')]

@dataclass
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    
    @staticmethod
    def dtype():
        return [('vx', 'f8'), ('vy', 'f8')]