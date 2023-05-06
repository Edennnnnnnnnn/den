"""
By Eden Zhou
Jan. 27, 2023
"""

import heapq
import numpy


class DijkstraSearch:
    def __init__(self, initState: State, goalState: State, spaceMap: Map):
        """
            This is the creator function which is used to initialize Dijkstra AI;
        :param:
            - initState, State, the initial position to start searching;
            - goalState, State, the target position to end searching;
            - spaceMap, Map, the game map applied for the searching task;
        """
        self.initState: State = initState
        self.goalState: State = goalState
        self.map: Map = spaceMap

        self.open: list = []
        self.close: dict = {}
        self.expansionsCounter: int = 0

    def search(self) -> tuple:
        """
            This is the searching function which is used to process Dijkstra AI;
        :return:
            - (cost: float, expansions: int), tuple, contains searching results: the searching costs (-1 for no-found),
        nun of node expanded in the searching process;
        """
        # Initialization:
        heapq.heappush(self.open, self.initState)
        self.close[self.initState.state_hash()] = self.initState

        while len(self.open) > 0:
            currState = heapq.heappop(self.open)
            self.expansionsCounter += 1
            # Stopping Condition:
            if currState == self.goalState:
                return currState.get_g(), self.expansionsCounter
            childrenValid = self.map.successors(currState)
            for child in childrenValid:
                childHashID = child.state_hash()

                # Encountering and Recording:
                if childHashID not in self.close.keys():
                    heapq.heappush(self.open, child)
                    self.close[childHashID] = child

                # Replacing and Updating:
                childPrev = self.close.get(childHashID)
                if childPrev is not None and child.get_g() < childPrev.get_g():
                    heapq.heappush(self.open, child)
                    self.close[childHashID] = child
                    # Re-heapify:
                    heapq.heapify(self.open)
        # If no-found:
        return -1, self.expansionsCounter
