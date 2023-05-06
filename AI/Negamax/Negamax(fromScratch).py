# Edited and adapted from CMPUT 455, codes of Martin Mueller

import time
import numpy as np

INFINITY = np.inf

def isBlackWhite(color):
    return (color == BLACK) or (color == WHITE)

def colorAsString(color):
    assert isBlackWhite(color)
    if color == BLACK:
        return "Black"
    else:
        return "White"
    
def opponent(color):
    assert isBlackWhite(color)
    return BLACK + WHITE - color


"""Boolean Version"""
""""""""""""""""""""""""""""""""""""
def negamaxBoolean(state):
    if state.endOfGame():
        return state.staticallyEvaluateForToPlay()
    for m in state.legalMoves():
        state.play(m)
        success = not negamaxBoolean(state)
        state.undoMove()
        if success:
            return True
    return False

def negamaxBooleanSolveAll(state):
    if state.endOfGame():
        return state.staticallyEvaluateForToPlay()
    wins = []
    for m in state.legalMoves():
        state.play(m)
        success = not negamaxBoolean(state)
        state.undoMove()
        if success:
            wins.append(m)
    return wins

def solveForColor(state, color): 
# use for 3-outcome games such as TicTacToe
    assert isBlackWhite(color)
    saveOldDrawWinner = state.drawWinner
    # to check if color can win, count all draws as win for opponent
    state.setDrawWinner(opponent(color)) 
    start = time.process_time()
    winForToPlay = negamaxBoolean(state)
    timeUsed = time.process_time() - start
    state.setDrawWinner(saveOldDrawWinner)
    winForColor = winForToPlay == (color == state.toPlay)
    return winForColor, timeUsed

def timed_solve(state): 
    start = time.process_time()
    wins = negamaxBooleanSolveAll(state)
    timeUsed = time.process_time() - start
    return wins, timeUsed


"""Naive Version"""
""""""""""""""""""""""""""""""""""""
def naive_negamax(state):
    if state.endOfGame():
        return state.staticallyEvaluateForToPlay()
    best = -INFINITY
    for m in state.legalMoves():
        state.play(m)
        value = -naive_negamax(state)
        if value > best:
            best = value
        state.undoMove()
    return best