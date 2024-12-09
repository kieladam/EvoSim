import numpy as np
from typing import Dict, List, Type, Set, Tuple
import uuid

class ComponentArray:
    def __init__(self, component_type: Type, initial_size: int = 1000):
        self.component_type = component_type
        self.size = initial_size
        self.data = np.zeros(initial_size, dtype=component_type.dtype())
        self.entity_to_index: Dict[uuid.UUID, int] = {}
        self.index_to_entity: Dict[int, uuid.UUID] = {}
        self.size_used = 0

    def insert(self, entity: uuid.UUID, component_data: dict):
        new_index = self.size_used
        self.entity_to_index[entity] = new_index
        self.index_to_entity[new_index] = entity
        
        for field, value in component_data.items():
            self.data[field][new_index] = value
            
        self.size_used += 1
        
        if self.size_used >= self.size:
            self._resize(self.size * 2)

    def remove(self, entity: uuid.UUID):
        index = self.entity_to_index[entity]
        last_index = self.size_used - 1
        
        if index != last_index:
            last_entity = self.index_to_entity[last_index]
            self.data[index] = self.data[last_index]
            
            self.entity_to_index[last_entity] = index
            self.index_to_entity[index] = last_entity
        
        del self.entity_to_index[entity]
        del self.index_to_entity[last_index]
        self.size_used -= 1

    def _resize(self, new_size: int):
        new_array = np.zeros(new_size, dtype=self.component_type.dtype())
        new_array[:self.size] = self.data
        self.data = new_array
        self.size = new_size

class View:
    def __init__(self, world: 'World', component_types: Tuple[Type, ...]):
        self.world = world
        self.component_types = component_types
        self._cached_entities = None
        self._last_update_count = -1

    def _update_cache(self):
        if self._last_update_count != self.world.update_count:
            # Find entities that have all required components
            matching_entities = set.intersection(
                *(set(self.world.components[comp_type].entity_to_index.keys())
                  for comp_type in self.component_types)
            )
            self._cached_entities = list(matching_entities)
            self._last_update_count = self.world.update_count

    def __iter__(self):
        self._update_cache()
        return iter(self._cached_entities)

    def __len__(self):
        self._update_cache()
        return len(self._cached_entities)

class World:
    def __init__(self):
        self.entities: Set[uuid.UUID] = set()
        self.components: Dict[Type, ComponentArray] = {}
        self.systems: List = []
        self.update_count = 0  # Track changes for view caching
        self._views: Dict[Tuple[Type, ...], View] = {}

    def create_entity(self) -> uuid.UUID:
        entity = uuid.uuid4()
        self.entities.add(entity)
        self.update_count += 1
        return entity

    def add_component(self, entity: uuid.UUID, component_type: Type, component_data: dict):
        if component_type not in self.components:
            self.components[component_type] = ComponentArray(component_type)
        self.components[component_type].insert(entity, component_data)
        self.update_count += 1

    def remove_component(self, entity: uuid.UUID, component_type: Type):
        self.components[component_type].remove(entity)
        self.update_count += 1

    def has_component(self, entity: uuid.UUID, component_type: Type) -> bool:
        if component_type not in self.components:
            return False
        return entity in self.components[component_type].entity_to_index

    def get_component(self, entity: uuid.UUID, component_type: Type):
        component_array = self.components[component_type]
        index = component_array.entity_to_index[entity]
        return component_array.data[index]

    def view(self, *component_types: Type) -> View:
        key = tuple(sorted(component_types, key=lambda x: id(x)))
        if key not in self._views:
            self._views[key] = View(self, key)
        return self._views[key]

    def add_system(self, system):
        self.systems.append(system)

    def remove_system(self, system):
        self.systems.remove(system)

    def update(self, delta):
        for system in self.systems:
            system.update(self, delta)
        self.update_count += 1