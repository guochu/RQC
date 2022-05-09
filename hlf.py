# -*- coding: utf-8 -*-
# @Author: guochu
# @Date:   2020-11-09 11:19:12
# @Last Modified by:   guochu
# @Last Modified time: 2020-11-11 21:34:31

# -*- coding: utf-8 -*-
# @Author: 1000787
# @Date:   2019-01-10 19:44:40
# @Last Modified by:   guochu
# @Last Modified time: 2020-11-09 09:19:19


from rqc.core import QuantumCircuit2D, generateQState2D, OneBodyObserver
from rqc.gates import CZ, S, H, Z

from math import sqrt

from numpy.random import randint
from numpy import zeros, asarray


import h5py
import time

def parse_cmd_line_args():
	args = sys.argv[1:]
	r = {}
	for arg in args:
		k, v = arg.split(":")
		r[k] = v
	return r


def toOneDIndex(idx, shape, major='R'):
	"""
	column major
	"""
	if major not in ['R', 'C']:
		raise ValueError('major must be R or C.')
	i, j = idx
	if not (i < shape[0] and j < shape[1]):
		raise IndexError('index out of range.')
	if major == 'R':
		return i*shape[1] + j
	else:
		return j*shape[0] + i

def toTwoDIndex(idx, shape, major='R'):
	"""
	column major
	"""
	if major not in ['R', 'C']:
		raise ValueError('major must be R or C.')
	if major == 'R':
		i = idx // shape[1]
		j = idx % shape[1]
		if i >= shape[0]:
			raise IndexError('index out of range.')
	else:
		i = idx % shape[0]
		j = idx // shape[0]
		if j >= shape[1]:
			raise IndexError('index out of range.')
	return i, j

def isHorV(l1, l2):
	if (l1[0] == l2[0] and l1[1]+1==l2[1]):
		return True
	if (l1[0] == l2[0] and l1[1]-1==l2[1]):
		return True
	if (l1[0]+1==l2[0] and l1[1]==l2[1]):
		return True
	if (l1[0]-1==l2[0] and l1[1]==l2[1]):
		return True
	return False

def generateA(n):
	# A = randint(low=0, high=2, size=(n, n))
	# A = A + A.T
	# for i in range(n):
	#     for j in range(n):
	#         if A[i, j]==2:
	#             A[i, j] = 0
	A = zeros((n, n), dtype=int)
	for i in range(n):
		for j in range(i+1, n):
			A[i, j] = randint(low=0, high=2)
	A = A + A.T
	for i in range(n):
		A[i, i] = randint(low=0, high=2)

	m = int(sqrt(n))
	assert(m*m == n)
	for i in range(n):
		l1 = toTwoDIndex(i, (m, m))
		for j in range(n):
			l2 = toTwoDIndex(j, (m, m))
			if isHorV(l1, l2)==False and i != j:
				A[i, j] = 0
	return A

def generate_circuit(A):
	assert(A.shape[0]==A.shape[1])
	n = A.shape[0]
	m = int(sqrt(n))
	assert(m*m == n)
	circuit = QuantumCircuit2D()
	for i in range(n):
		l1 = toTwoDIndex(i, (m, m))
		circuit.append((l1, H))
	for i in range(n):
		l1 = toTwoDIndex(i, (m, m))
		for j in range(i, n):
			l2 = toTwoDIndex(j, (m, m))
			if i==j:
				if A[i, j] == 1:
					circuit.append((l1, S))
			elif isHorV(l1, l2):
				if A[i, j] == 1:
					circuit.append(((l1, l2), CZ))
			else:
				assert(A[i, j] == 0)

	for i in range(n):
		l1 = toTwoDIndex(i, (m, m))
		circuit.append((l1, H))
	# for i in range(n-1):
	# 	observer = ClassicalObserver2D([toTwoDIndex(i, (m, m)), toTwoDIndex(i+1, (m, m))], 'XY')
	# 	circuit.append(observer)
	# circuit.append(AmplitudeObserver2D([[0]*m]*m))
	for i in range(n):
		observer = OneBodyObserver(toTwoDIndex(i, (m, m)))
		circuit.append(observer)
	return circuit

def to_matrix(s, shape):
	a = zeros(shape, dtype=int)
	for i in range(len(s)):
		a[toTwoDIndex(i, shape)] = s[i]
	return a

if __name__ == '__main__':
	# paras = parse_cmd_line_args()

	# n = int(paras['n'])
	n = 8

	n2 = n * n
	A = generateA(n2)
	# print(A)


	circuit = generate_circuit(A)
	mps = generateQState2D(zeros((n, n), dtype=int))
	result = circuit.run(mps)

	# result = circuit.collect()
	r = [value for (name, value) in result]
	print(r)

