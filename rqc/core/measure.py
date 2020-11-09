# -*- coding: utf-8 -*-
# @Author: guochu
# @Date:   2020-11-09 10:55:24
# @Last Modified by:   guochu
# @Last Modified time: 2020-11-09 11:23:08
from numpy.random import uniform
from numpy import sqrt
from rqc.tensor import astensor 
from .circuit2d import OneBodyGate

__all__ = ['OneBodyObserver']

def _discrete_sample(l):
	s = sum(l)
	# print('sum is', s)
	if (abs(s) < 1.0e-12):
		# print('s is', s)
		raise ValueError('error in measure.')
	l1 = [None]*(len(l)+1)
	l1[0] = 0.
	for i in range(len(l)):
		l1[i+1] = l1[i] + l[i]/s
	s = uniform(low=0., high=1.)
	for i in range(len(l1)-1):
		if (s >= l1[i] and s < l1[i+1]):
			return i
	raise NotImplementedError('should not be here.')


class OneBodyObserver:
	"""docstring for OneBodyObserver"""
	def __init__(self, key):
		if len(key) != 2:
			raise ValueError('wrong position for one body gate.')
		for s in key:
			if not isinstance(s, int):
				raise TypeError('position must be integer type.')
		self.key = tuple(key)

	def apply(self, state, maxbonddimension=2000, svdcutoff=1.0e-8, verbose=1):
		up = astensor([[1., 0.], [0., 0.]])
		down = astensor([[0., 0.], [0., 1.]])
		up_gate = OneBodyGate(self.key, up)
		# ms = close_peps(state, state, {self.key:up})
		state_1 = state.copy()
		up_gate.apply(state, maxbonddimension=maxbonddimension, svdcutoff=svdcutoff, verbose=verbose)
		s = state_1.cross(state, conj=True, scale_factor=1.)
		tol = 1.0e-6
		if s.imag > tol:
			warnings.warn('the imaginary part of result is larger than'+str(tol))
		s = abs(s.real)
		l = [s, abs(1-s)]
		i = _discrete_sample(l)
		# print('probability is', s, 'istate is', i)
		if i==0:
			state[self.key] = up.contract(state[self.key], ((1,), (0,)))
			state[self.key] /= sqrt(l[i])
		else:
			state[self.key] = down.contract(state[self.key], ((1,), (0,)))
			state[self.key] /= sqrt(l[i])		
		self.istate = i
		self.probability = l[i]

	def apply_and_collect(self, state, result, maxbonddimension=2000, svdcutoff=1.0e-8, verbose=1):
		self.apply(state, maxbonddimension, svdcutoff, verbose=verbose)
		result.append(self.result)


	@property
	def name(self):
		return 'Q:Z' + str(self.key)

	@property
	def result(self):
		return (self.name, self.istate)