# Edited and adapted from CMPUT 455, codes of Martin Mueller

import numpy as np

INFINITY = np.inf

SPACES_PER_DEPTH = 4
def indent(depth):
		return ' ' * (SPACES_PER_DEPTH * depth)


"""Naive Version"""
""""""""""""""""""""""""""""""""""""
def alphabeta(state, alpha, beta, depth):
		print(f"{indent(depth)} Call alphabeta with window ({alpha}, {beta})")
		if state.endOfGame():
				v = state.staticallyEvaluateForToPlay() 
				print(f"{indent(depth)} Leaf node value {v}")
				return v
		for m in state.legalMoves():
				print(f"{indent(depth)} play {m}")
				state.play(m)
				value = -alphabeta(state, -beta, -alpha, depth + 1)
				if value > alpha:
						alpha = value
						print(f"{indent(depth)} New best value {value}, new window ({alpha}, {beta})")
				print(f"{indent(depth)} undo {m}")
				state.undoMove()
				if value >= beta: 
						print(f"{indent(depth)} beta cut {value} >= {beta}, return {beta}")
						return beta   # or value in failsoft (later)
		print(f"{indent(depth)} tried all moves, return best value {alpha}")
		return alpha
	
	
"""Depth-limited Heuristic Version"""
""""""""""""""""""""""""""""""""""""
def alphabetaDL(state, alpha, beta, depth):
	if state.endOfGame() or depth == 0:
		return state.staticallyEvaluateForToPlay() 
	for m in state.legalMoves():
		state.play(m)
		value = -alphabetaDL(state, -beta, -alpha, depth - 1)
		if value > alpha:
			alpha = value
		state.undoMove()
		if value >= beta: 
			return beta   # or value in failsoft (later)
	return alpha

# initial call with full window
def callAlphabeta(rootState):
		return alphabeta(rootState, -INFINITY, INFINITY, 0)
	
	