import numpy as np
""" 
A SumTree is an efficient data structure for storing and sampling transitions based on their priority using a binary tree, 
where parent nodes store the sum of their children's priorities.

The leaf nodes represent individual transition priorities, while internal nodes store 
cumulative sums for fast retrieval. This allows for sampling and updating priorities in O(log n) time.

Our implementation is largely based on this blog post:
https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
Also helpful: 
https://www.sefidian.com/2022/11/09/sumtree-data-structure-for-prioritized-experience-replay-per-explained-with-python-code/
"""
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=np.int32)
        self._data_pointer = 0
        
    def _propagate(self, idx, change):
        """Propagate a change in priority through the sum tree.

        Args:
            idx (int): The current index in the sum tree.
            change (float): The change in priority to propagate.
        """
        parent_idx = (idx - 1) // 2
        self.tree[parent_idx] += change
        
        if parent_idx != 0:
            self._propagate(parent_idx, change)
            
    def _retrieve(self, idx, s):
        """Retrieve the index of a transition based on a sum value.

        Args:
            idx (int): The current index in the sum tree.
            s (float): The sum value to retrieve the index for.

        Returns:
            _type_: _description_
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return the total sum of priorities stored in the sum tree.

        Returns:
            float: The total sum of priorities.
        """
        return self.tree[0]
            
    def update(self, idx, priority):
        """Update the priority of a transition and propagate the change through the tree.

        Args:
            idx (int): The index of the transition in the sum tree.
            priority (float): The new priority of the transition.
        """
    
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        
        self._propagate(idx, change)
        
    
    def add(self, priority, data):
        """Add a new transition to the sum tree.

        Args:
            priority (float): The priority of the new transition.
            data (object): The data associated with the transition.
        """
        leaf_idx = self._data_pointer + self.capacity - 1 
        
        self.data[self._data_pointer] = data
        self.update(leaf_idx, priority)
        
        self._data_pointer = (self._data_pointer + 1) % self.capacity 
        
    def get_leaf(self, v):
        """Get the leaf index, priority, and data of a transition based on a value.

        Args:
            v (float): The random value to retrieve the transition for.

        Returns:
            Tuple[int, float, object]: 
                - The leaf index in the sum tree.
                - The priority of the transition.
                - The stored data associated with the transition.
        """
        leaf_idx = self._retrieve(0, v)
        data_idx = leaf_idx - self.capacity + 1
       
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

        
        