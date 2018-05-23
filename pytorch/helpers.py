import numpy as np
import torch
from time import time
import cProfile, pstats

def check_correctness(full, names, approximations):
	print("Checking correctness")
	
	true_value = full()
	approx_values = list()
	for i in range(len(approximations)):
		approx_value = approximations[i]()
		approx_values.append(approx_value)
		print("Error on approximation %d (%5s): %f" % (i, names[i], np.linalg.norm(true_value - approx_value)))

def simpleTiming(full, names, approximations, REPEATS=10):
	print("Simple timing")
	
	def timeRun(method):
		start = time()
		for r in range(REPEATS):
			method()
		end = time()
		return (end - start)/REPEATS
		
	print("Full  : %f" % timeRun(full))
	for i in range(len(approximations)):
		print("(%5s): %f" % (names[i], timeRun(approximations[i])))

def profiling(full, names, approximations, REPEATS=1, Prec=20):
	print("Profiling")

	def profile(method):
		pr = cProfile.Profile()
		pr.enable()
		for r in range(REPEATS):
			method()
		pr.disable()
		pr.create_stats()
		ps = pstats.Stats(pr).sort_stats("cumulative")
		ps.print_stats(Prec)
		
	print("Full:")
	profile(full)
	for i in range(len(approximations)):
		print(names[i])
		profile(approximations[i])
