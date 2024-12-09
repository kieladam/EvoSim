import ECS as e
import ECScomponents as ec
import ECSsystems as es

# Example usage with performance comparison
def main():
    import time
    delta = 0.0
    world = e.World()
    
    # Create many entities, but only some with both components
    num_entities = 1000000
    num_with_both = 100000
    
    for i in range(num_entities):
        entity = world.create_entity()
        world.add_component(entity, ec.Position, {'x': 0.0, 'y': 0.0})
        if i < num_with_both:  # Only first 1000 entities get velocity
            world.add_component(entity, ec.Velocity, {'vx': 1.0, 'vy': 2.0})
    
    world.add_system(es.MovementSystem())
    
    start_time = time.time()
    for _ in range(100):  # 100 updates
        world.update(delta)
    end_time = time.time()
    
    print(f"Processed {num_entities} entities ({num_with_both} with both components)")
    print(f"Time taken for 100 updates: {end_time - start_time:.4f} seconds")

main()