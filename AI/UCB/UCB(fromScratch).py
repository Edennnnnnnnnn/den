# Edited and adapted from CMPUT 455, codes of Martin Mueller

from bernoulli import bernoulliExperiment
from math import sqrt, log
import numpy as np

INFINITY = np.inf

def defaultInit(stats, n):
	stats = [[0,0] for _ in range(n)]
	
def simulateEasy(i): # Easy case: payoffs 0.0, 0.1,...,0.9
	assert i >= 0
	assert i < 10
	return bernoulliExperiment(i/10)

def simulateHard(i): # Hard case: arm 1 and 2 very close. Best arm: 2
	payoff = [0.5, 0.61, 0.62, 0.55]
	assert i >= 0
	assert i < 4
	return bernoulliExperiment(payoff[i])

def mean(stats, i):
	return stats[i][0] / stats[i][1]

def ucb(stats, C, i, n):
	if stats[i][1] == 0:
		return INFINITY
	return mean(stats, i)  + C * sqrt(log(n) / stats[i][1])

def findBest(stats, C, n):
	best = -1
	bestScore = -INFINITY
	for i in range(len(stats)):
		score = ucb(stats, C, i, n) 
		if score > bestScore:
			bestScore = score
			best = i
	assert best != -1
	return best

def bestArm(stats): # Most-pulled arm
	best = -1
	bestScore = -INFINITY
	for i in range(len(stats)):
		if stats[i][1] > bestScore:
			bestScore = stats[i][1]
			best = i
	assert best != -1
	return best

def runUcb(C, arms, init, simulate, maxSimulations):
	stats = [[0,0] for _ in range(arms)]
	for n in range(maxSimulations):
		move = findBest(stats, C, n)
		if simulate(move):
			stats[move][0] += 1 # win
		stats[move][1] += 1
	print("C = {} Statistics: {} Best arm {}".format(C, stats, bestArm(stats)))
	
for C in [100, 10, sqrt(2), 1, 0.1, 0.01]:
	runUcb(C, 10, defaultInit, simulateEasy, 1000)
	#runUcb(C, 4, defaultInit, simulateHard, 1000)
	#runUcb(C, 4, defaultInit, simulateHard, 100000)