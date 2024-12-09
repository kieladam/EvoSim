import ECS as e
import ECScomponents as ec

# Example Movement System using View
class MovementSystem:
    def update(self, world: e.World, delta):
        # Get view of entities with both Position and Velocity
        view = world.view(ec.Position, ec.Velocity)
        
        pos_array = world.components[ec.Position]
        vel_array = world.components[ec.Velocity]

        # Process only entities that have both components
        for entity in view:
            pos_idx = pos_array.entity_to_index[entity]
            vel_idx = vel_array.entity_to_index[entity]
            
            # Update position based on velocity
            pos_array.data['x'][pos_idx] += vel_array.data['vx'][vel_idx]
            pos_array.data['y'][pos_idx] += vel_array.data['vy'][vel_idx]