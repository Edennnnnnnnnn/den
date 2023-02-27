"""
By Eden Zhou
Jan. 27, 2023
"""

class BiBS:
    def __init__(self, initState: State, goalState: State, spaceMap: Map):
        """
            This is the creator function which is used to initialize Bidirectional Brute-force Search;
        :param:
            - initState, State, the initial position to start searching;
            - goalState, State, the target position to end searching;
            - spaceMap, Map, the game map applied for the searching task;
        """
        self.initState: State = initState
        self.goalState: State = goalState
        self.map: Map = spaceMap

        self.openF: list = []
        self.closeF: dict = {}
        self.expandingCounterF: int = 0
        self.openB: list = []
        self.closeB: dict = {}
        self.expandingCounterB: int = 0
        self.U: float = numpy.inf

    def search(self) -> tuple:
        """
            This is the searching function which is used to process Bidirectional Brute-force Search;
        :return:
            - (cost: float, expansions: int), tuple, contains searching results: the searching costs (-1 for no-found),
        nun of node expanded in the searching process;
        """
        # Initialization:
        heapq.heappush(self.openF, self.initState)
        self.closeF[self.initState.state_hash()] = self.initState
        heapq.heappush(self.openB, self.goalState)
        self.closeB[self.goalState.state_hash()] = self.goalState

        while len(self.openF) > 0 and len(self.openB) > 0:
            # Stopping Condition:
            if self.U <= self._getUnionCost(self.openF[0], self.openB[0]):
                return self.U, self._getUnionExpansions(self.expandingCounterF, self.expandingCounterB)
            # Expanding Forward Search:
            if self.openF[0] < self.openB[0]:
                currState = heapq.heappop(self.openF)
                self.expandingCounterF += 1
                childrenValid = self.map.successors(currState)
                for child in childrenValid:
                    childHashID = child.state_hash()

                    # Computing The Cost of Solution Path (going through n′ if n′ is in CLOSED_b)
                    if childHashID in self.closeB.keys():
                        self.U = min(self.U, self._getUnionCost(child, self.closeB.get(childHashID)))

                    # Encountering and Recording:
                    if childHashID not in self.closeF.keys():
                        heapq.heappush(self.openF, child)
                        self.closeF[childHashID] = child

                    # Replacing and Updating:
                    childPrev = self.closeF.get(childHashID)
                    if childPrev is not None and child.get_g() < childPrev.get_g():
                        heapq.heappush(self.openF, child)
                        self.closeF[childHashID] = child
                        # Re-heapify:
                        heapq.heapify(self.openF)

            # Expanding Backward Search (exactly as above but with OPEN_b)
            else:
                currState = heapq.heappop(self.openB)
                self.expandingCounterB += 1
                childrenValid = self.map.successors(currState)
                for child in childrenValid:
                    childHashID = child.state_hash()

                    # Computing The Cost of Solution Path (going through n′ if n′ is in CLOSED_b)
                    if childHashID in self.closeF.keys():
                        self.U = min(self.U, self._getUnionCost(child, self.closeF.get(childHashID)))

                    # Encountering and Recording:
                    if childHashID not in self.closeB.keys():
                        heapq.heappush(self.openB, child)
                        self.closeB[childHashID] = child

                    # Replacing and Updating:
                    childPrev = self.closeB.get(childHashID)
                    if childPrev is not None and child.get_g() < childPrev.get_g():
                        heapq.heappush(self.openB, child)
                        self.closeB[childHashID] = child
                        # Re-heapify:
                        heapq.heapify(self.openB)
        # If no-found:
        return -1, self._getUnionExpansions(self.expandingCounterF, self.expandingCounterB)

    @staticmethod
    def _getUnionCost(stateF: State, stateB: State) -> float:
        """
            This is a helper function used to compute the union cost summation of two processes (b/w two states);
        :param:
            - stateF, State, the first state (from the forward process), entered for computing the union cost;
            - stateB, State, the second state (from the backward process), entered for computing the union cost;
        """
        return stateF.get_g() + stateB.get_g()

    @staticmethod
    def _getUnionExpansions(expansionsF: int, expansionsB: int) -> int:
        """
            This is a helper function used to compute the union expansions summation of two processes;
        :param:
            - expansionsF, int, the first expansion num (from the forward process), used for computing the union expansions;
            - expansionsB, int, the second expansion num (from the backward process), used for computing the union expansions;
        """
        return expansionsF + expansionsB

    