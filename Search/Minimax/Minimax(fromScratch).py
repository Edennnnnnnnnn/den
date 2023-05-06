# Edited and adapted from CMPUT 455, codes of Martin Mueller

import time
import numpy as np


EMPTY = 0 
BLACK = 1 # Also used for 'X'
WHITE = 2 # Also used for 'O'

INFINITY = np.inf

def opponent(color):
    assert isBlackWhite(color)
    return BLACK + WHITE - color



"""Boolean Version"""
""""""""""""""""""""""""""""""""""""
def minimaxBooleanOR(state):
    assert state.toPlay == BLACK
    if state.endOfGame():
        return state.isWinner(BLACK)
    for m in state.legalMoves():
        state.play(m)
        isWin = minimaxBooleanAND(state)
        state.undoMove()
        if isWin:
            return True
    return False

def minimaxBooleanAND(state):
    assert state.toPlay == WHITE
    if state.endOfGame():
        return state.isWinner(BLACK)
    for m in state.legalMoves():
        state.play(m)
        isLoss = not minimaxBooleanOR(state)
        state.undoMove()
        if isLoss:
            return False
    return True

def solveForBlack(state): 
    win = False
    start = time.process_time()
    if state.toPlay == BLACK:
        win = minimaxBooleanOR(state)
    else:
        win = minimaxBooleanAND(state)
    timeUsed = time.process_time() - start
    return win, timeUsed


"""Naive Version"""
""""""""""""""""""""""""""""""""""""
def minimaxOR(state):
    if state.endOfGame():
        return state.staticallyEvaluate() 
    best = -INFINITY
    for m in state.legalMoves():
        state.play(m)
        value = minimaxAND(state)
        if value > best:
            best = value
        state.undoMove()
    return best

def minimaxAND(state):
    if state.endOfGame():
        return state.staticallyEvaluate() 
    best = INFINITY
    for m in state.legalMoves():
        state.play(m)
        value = minimaxOR(state)
        if value < best:
            best = value
        state.undoMove()
    return best