"""
By Eden Zhou
Feb. 28, 2023
"""

import heapq
import numpy


class AStar:
    def __init__(self, initState: State, goalState: State, spaceMap: Map):
        """
            This is the creator function which is used to initialize AStar AI;
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
            This is the searching function which is used to process AStar AI;
        :return:
            - (cost: float, expansions: int), tuple, contains searching results: the searching costs (-1 for no-found),
        nun of node expanded in the searching process;
        """
        # Initialization:
        heapq.heappush(self.open, self.initState)
        self.close[self.initState.state_hash()] = self.initState
        self.initState.set_cost(self._getFValue(self.initState, self.goalState))

        while len(self.open) > 0:
            currState = heapq.heappop(self.open)
            # Stopping Condition:
            if currState == self.goalState:
                return currState.get_cost(), self.expansionsCounter
            self.expansionsCounter += 1
            childrenValid = self.map.successors(currState)
            for child in childrenValid:
                # Computing the hash key and the f-value (g-value + h-value) for each node:
                childHashID = child.state_hash()
                child.set_cost(self._getFValue(child, self.goalState))

                # Encountering and Recording:
                if childHashID not in self.close.keys():
                    heapq.heappush(self.open, child)
                    self.close[childHashID] = child

                # Replacing and Updating:
                childPrev = self.close.get(childHashID)
                if childPrev is not None and child.get_g() < childPrev.get_g():
                    self.open[self.open.index(childPrev)] = child
                    self.close[childHashID] = child
                    # Re-heapify:
                    heapq.heapify(self.open)


        # If no-found:
        return -1, self.expansionsCounter

    def _getFValue(self, nodeLocal: State, nodeGoal: State) -> float:
        """
            This is a helper function which is used to compute the F-value based on the given information;
        :params:
            - nodeLocal, State, the current agent location;
            - nodeGoal, State, the target location in searching;
        :returns:
            - fValue, float, the F-value computed based on the given information, where f(x) = GhValue + H-Value (Octile
        Distance based);
        """
        fValue = nodeLocal.get_g() + self._getOctileDistance(nodeLocal, nodeGoal)
        return fValue

    @staticmethod
    def _getOctileDistance(nodeLocal: State, nodeGoal: State) -> float:
        """
            This is a helper function which is used to compute the Octile Distance based on two given states;
        :params:
            - nodeLocal, State, the current agent location;
            - nodeGoal, State, the target location in searching;
        :returns:
            - octileDistance, float, the Octile Distance between two given points, where the octile distance is
        computed by the formula of 1.5·min(|X_a - X_b|, |Y_a - Y_b|) + ||X_a - X_b| - |Y_a - Y_b||;
        """
        delta_x = abs(nodeLocal.get_x() - nodeGoal.get_x())
        delta_y = abs(nodeLocal.get_y() - nodeGoal.get_y())

        return 1.5 * min(delta_x, delta_y) + abs(delta_x - delta_y)