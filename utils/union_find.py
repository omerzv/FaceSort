"""Union-Find (Disjoint Set) data structure for robust cluster merging."""

from typing import Dict, List, Set, Tuple


class UnionFind:
    """Union-Find data structure with path compression and union by rank."""
    
    def __init__(self, elements: List[int]) -> None:
        """Initialize Union-Find with given elements.
        
        Args:
            elements: List of cluster IDs to initialize
        """
        self.parent: Dict[int, int] = {elem: elem for elem in elements}
        self.rank: Dict[int, int] = {elem: 0 for elem in elements}
        self.size: Dict[int, int] = {elem: 1 for elem in elements}
    
    def find(self, x: int) -> int:
        """Find root of element with path compression.
        
        Args:
            x: Element to find root for
            
        Returns:
            Root element
        """
        if self.parent[x] != x:
            # Path compression: make parent point directly to root
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union two elements by rank.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if union was performed, False if already connected
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank: attach smaller tree to root of larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
        
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in the same set.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if connected, False otherwise
        """
        return self.find(x) == self.find(y)
    
    def get_components(self) -> Dict[int, List[int]]:
        """Get all connected components.
        
        Returns:
            Dictionary mapping root -> list of elements in component
        """
        components: Dict[int, List[int]] = {}
        
        for element in self.parent.keys():
            root = self.find(element)
            if root not in components:
                components[root] = []
            components[root].append(element)
        
        return components
    
    def get_component_sizes(self) -> Dict[int, int]:
        """Get sizes of all components.
        
        Returns:
            Dictionary mapping root -> component size
        """
        sizes = {}
        for element in self.parent.keys():
            root = self.find(element)
            sizes[root] = self.size[root]
        return sizes