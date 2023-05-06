"""
By Eden Zhou
Feb. 28, 2023
"""

import heapq
import numpy

class BiAStar:
	def __init__(self, initState: State, goalState: State, spaceMap: Map):
		"""
			This is the creator function which is used to initialize Bidirectional AStar AI;
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
		self.openB: list = []
		self.closeB: dict = {}
		
		self.U: float = numpy.inf
		self.expandingCounter: int = 0
		
	def search(self) -> tuple:
		"""
			This is the searching function which is used to process Bidirectional AStar AI;
		:return:
			- (cost: float, expansions: int), tuple, contains searching results: the searching costs (-1 for no-found),
		nun of node expanded in the searching process;
		"""
		# Initialization:
		self.initState.set_cost(self._getFValue(self.initState, self.goalState))
		self.goalState.set_cost(self._getFValue(self.goalState, self.initState))
		
		heapq.heappush(self.openF, self.initState)
		self.closeF[self.initState.state_hash()] = self.initState
		heapq.heappush(self.openB, self.goalState)
		self.closeB[self.goalState.state_hash()] = self.goalState
		
		while len(self.openF) > 0 and len(self.openB) > 0:
			# Stopping Condition:
			if self.U <= min(self.openF[0].get_cost(), self.openB[0].get_cost()):
				return self.U, self.expandingCounter
			# Expanding Forward AI:
			if self.openF[0] < self.openB[0]:
				currState = heapq.heappop(self.openF)
				self.expandingCounter += 1
				childrenValid = self.map.successors(currState)
				for child in childrenValid:
					# Computing the hash key and the f-value (g-value + h-value) for each node:
					childHashID = child.state_hash()
					child.set_cost(self._getFValue(child, self.goalState))
					
					# Computing The Cost of Solution Path (going through n′ if n′ is in CLOSED_b)
					if childHashID in self.closeB.keys():
						self.U = min(self.U, self._getUnionCost(child, self.closeB.get(childHashID)))
						
					# Encountering and Recording:
					if childHashID not in self.closeF.keys():
						heapq.heappush(self.openF, child)
						self.closeF[childHashID] = child
						
					# Replacing and Updating:
					childPrev = self.closeF.get(childHashID)
					if childPrev is not None and child.get_cost() < childPrev.get_cost():
						self.openF[self.openF.index(childPrev)] = child
						self.closeF[childHashID] = child
						# Re-heapify:
						heapq.heapify(self.openF)
						
			# Expanding Backward AI (exactly as above but with OPEN_b)
			else:
				currState = heapq.heappop(self.openB)
				self.expandingCounter += 1
				childrenValid = self.map.successors(currState)
				for child in childrenValid:
					# Computing the hash key and the f-value (g-value + h-value) for each node:
					childHashID = child.state_hash()
					child.set_cost(self._getFValue(child, self.initState))
					
					# Computing The Cost of Solution Path (going through n′ if n′ is in CLOSED_b)
					if childHashID in self.closeF.keys():
						self.U = min(self.U, self._getUnionCost(child, self.closeF.get(childHashID)))
						
					# Encountering and Recording:
					if childHashID not in self.closeB.keys():
						heapq.heappush(self.openB, child)
						self.closeB[childHashID] = child
						
					# Replacing and Updating:
					childPrev = self.closeB.get(childHashID)
					if childPrev is not None and child.get_cost() < childPrev.get_cost():
						self.openB[self.openB.index(childPrev)] = child
						self.closeB[childHashID] = child
						# Re-heapify:
						heapq.heapify(self.openB)
		# If no-found:
		return -1, self.expandingCounter
	
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
	
	@staticmethod
	def _getUnionCost(stateF: State, stateB: State) -> float:
		"""
			This is a helper function used to compute the union cost summation of two processes (b/w two states);
		:param:
			- stateF, State, the first state (from the forward process), entered for computing the union cost;
			- stateB, State, the second state (from the backward process), entered for computing the union cost;
		:return:
			- overallCost, float;
		"""
		return stateF.get_g() + stateB.get_g()