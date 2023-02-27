class State:
    """
    Class to represent a state on grid-based pathfinding problems. The class contains a static variable:
    map_width containing the width of the map. Although this variable is a property of the map and not of 
    the state, the property is used to compute the hash value of the state, which is used in the CLOSED list. 

    Each state has the values of x, y, g, and cost. The cost is used as the criterion for sorting the nodes
    in the OPEN list for A*, Bi-A*, and MM. For A* and Bi-A* the cost should be the f-value of the node, while
    for MM the cost should be the p-value of the node. 
    """
    map_width = 0
    
    def __init__(self, x, y):
        """
        Constructor - requires the values of x and y of the state. All the other variables are
        initialized with the value of 0.
        """
        self._x = x
        self._y = y
        self._g = 0
        self._cost = 0
        
    def __repr__(self):
        """
        This method is invoked when we call a print instruction with a state. It will print [x, y],
        where x and y are the coordinates of the state on the map. 
        """
        state_str = "[" + str(self._x) + ", " + str(self._y) + "]"
        return state_str
    
    def __lt__(self, other):
        """
        Less-than operator; used to sort the nodes in the OPEN list
        """
        return self._cost < other._cost
    
    def state_hash(self):
        """
        Given a state (x, y), this method returns the value of x * map_width + y. This is a perfect 
        hash function for the problem (i.e., no two states will have the same hash value). This function
        is used to implement the CLOSED list of the algorithms. 
        """
        return self._y * State.map_width + self._x
    
    def __eq__(self, other):
        """
        Method that is invoked if we use the operator == for states. It returns True if self and other
        represent the same state; it returns False otherwise. 
        """
        return self._x == other._x and self._y == other._y

    def get_x(self):
        """
        Returns the x coordinate of the state
        """
        return self._x
    
    def get_y(self):
        """
        Returns the y coordinate of the state
        """
        return self._y
    
    def get_g(self):
        """
        Returns the g-value of the state
        """
        return self._g
        
    def set_g(self, cost):
        """
        Sets the g-value of the state
        """
        self._g = cost

    def get_cost(self):
        """
        Returns the g-value of the state
        """
        return self._cost
        
    def set_cost(self, cost):
        """
        Sets the g-value of the state
        """
        self._cost = cost
