import time
from dataclasses import dataclass
from typing import List, Dict
import uuid

@dataclass
class GameObject:
    id: uuid.UUID
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    has_velocity: bool = False
    
    def update(self):
        """Update the object's position based on its velocity"""
        if self.has_velocity:
            self.x += self.vx
            self.y += self.vy

class GameWorld:
    def __init__(self):
        self.objects: Dict[uuid.UUID, GameObject] = {}
    
    def create_static_object(self, x: float = 0.0, y: float = 0.0) -> uuid.UUID:
        obj_id = uuid.uuid4()
        self.objects[obj_id] = GameObject(
            id=obj_id,
            x=x,
            y=y,
            has_velocity=False
        )
        return obj_id
    
    def create_moving_object(self, x: float = 0.0, y: float = 0.0, 
                           vx: float = 0.0, vy: float = 0.0) -> uuid.UUID:
        obj_id = uuid.uuid4()
        self.objects[obj_id] = GameObject(
            id=obj_id,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            has_velocity=True
        )
        return obj_id
    
    def update(self):
        """Update all objects in the world"""
        for obj in self.objects.values():
            obj.update()

def main():
    # Create OOP world
    world = GameWorld()
    
    # Create same number of objects as ECS example
    num_entities = 1000000
    num_with_both = 100000
    
    # Create static objects (position only)
    for _ in range(num_entities - num_with_both):
        world.create_static_object(x=0.0, y=0.0)
    
    # Create moving objects (position and velocity)
    for _ in range(num_with_both):
        world.create_moving_object(x=0.0, y=0.0, vx=1.0, vy=2.0)
    
    # Run performance test
    start_time = time.time()
    for _ in range(100):  # 100 updates
        world.update()
    end_time = time.time()
    
    print(f"OOP Implementation (with object methods):")
    print(f"Processed {num_entities} objects ({num_with_both} with velocity)")
    print(f"Time taken for 100 updates: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()