
import warnings
from numpy.random import rand
from numpy import zeros, array, sqrt

from rqc.tensor import astensor
from .overlap import rescaled_overlap
# import json

__add__ = ['PEPS', 'createrandompeps', 'generateProdPEPS']

class PEPS:
	"""two dimensional tensor network.
	The index convention for single site
	peps: ------------1-------------
	--------------2---0---3---------
	------------------4-------------
	"""
	def __init__(self, shape):
		# assert(len(shape)==2)
		if len(shape) != 2:
			raise ValueError('error input shape.')
		self.shape = shape
		self.data = [None]*self.size()
		# self.svectors = [None]*((self.m+1)*(self.n+1))

	def __iter__(self):
		return self.data

	def size(self):
		return self.shape[0]*self.shape[1]

	def __sgi(self, p):
		i, j = p
		# assert(i < self.shape[0] and j < self.shape[1])
		if not (i < self.shape[0] and j < self.shape[1]):
			raise IndexError('index out of bound.')
		return j*self.shape[0]+i

	def __getitem__(self, key):
		return self.data[self.__sgi(key)]

	def __setitem__(self, key, value):
		self.data[self.__sgi(key)] = astensor(value)

	def copy(self):
		r = type(self)(self.m, self.n)
		for i in range(self.m):
			for j in range(self.n):
				r[(i, j)] = self[(i, j)].copy()
		return r

	def conj(self):
		r = type(self)(self.m, self.n)
		for i in range(self.m):
			for j in range(self.n):
				r[(i, j)] = self[(i, j)].conj()
		return r

	def cross(self, other, conj=True, scale_factor=sqrt(2)):
		if not isinstance(other, type(self)):
			raise TypeError('wrong type for other.')
		if conj==True:
			s = rescaled_overlap(self, other, scale_factor)
		else:
			s = rescaled_overlap(self.conj(), other, scale_factor)
		return s


	def bond_dimension(self):
		return self.bond_dimensions().max()

	def bond_dimensions(self):
		r = zeros((self.shape[0]-1, self.shape[1]-1))
		for i in range(self.shape[0]-1):
			for j in range(self.shape[1]-1):
				s = self[(i, j)].shape
				r[i, j] = max([s[1],s[2],s[3],s[4]])
		return r

	def __str__(self):
		# ss = str()
		# for i in range(self.shape[0]):
		# 	for j in range(self.shape[1]):
		# 		ss += 'peps on sites' + str((i, j)) + '\n'
		# 		ss += self[(i, j)].__str__()
		# return ss
		return 'Two dimensional quantum state'


# def savetxt(peps, file_name):
# 	data_real = [m.real().tolist() for m in peps.data]
# 	data_imag = [m.imag().tolist() for m in peps.data]
# 	with open(file_name, 'w') as f:
# 		json.dump({'shape':peps.shape, 'data_real':data_real, 'data_imag':data_imag}, f)

# def loadtxt(file_name):
# 	with open(file_name, 'r') as f:
# 		obj = json.load(f)
# 	r = PEPS(obj['shape'])
# 	data_real = obj['data_real']
# 	data_imag = obj['data_imag']
# 	for i in range(r.shape[0]):
# 		for j in range(r.shape[1]):
# 			r[(i,j)]=astensor(array(data_real[j*r.shape[0]+i]) + 1j*array(data_imag[j*r.shape[0]+i]))
# 	return r

def createrandompeps(shape, d, D):
	r = PEPS(shape)
	r[(0,0)] = rand(d, 1, 1, D, D)
	r[(r.shape[0]-1, 0)] = rand(d, D, 1, D, 1)
	r[(0,r.shape[1]-1)] = rand(d, 1, D, 1, D)
	r[(r.shape[0]-1, r.shape[1]-1)] = rand(d, D, D, 1, 1)
	for i in range(1, r.shape[0]-1):
		r[(i, 0)] = rand(d, D, 1, D, D)
		r[(i, r.shape[1]-1)] = rand(d, D, D, 1, D)
	for i in range(1, r.shape[1]-1):
		r[(0, i)] = rand(d, 1, D, D, D)
		r[(r.shape[0]-1, i)] = rand(d, D, D, D, 1)
	for i in range(1,r.shape[0]-1):
		for j in range(1,r.shape[1]-1):
			r[(i, j)] = rand(d, D, D, D, D)
	return r

def generateProdPEPS(shape, ds, mpsstr):
	# assert(len(ds) == len(mpsstr))
	# assert(len(ds) == shape[0])
	if not (len(ds) == len(mpsstr) and len(ds) == shape[0]):
		raise ValueError('input shape mismatch.')
	for i in range(shape[0]):
		# assert(len(ds[i]) == shape[1])
		# assert(len(mpsstr[i]) == shape[1])
		if not (len(ds[i]) == shape[1] and len(mpsstr[i]) == shape[1]):
			raise ValueError('input shape mismatch.')
	peps = PEPS(shape)
	for i in range(shape[0]):
		for j in range(shape[1]):
			d = ds[i][j]
			peps[(i, j)] = zeros((d,1,1,1,1))
			if hasattr(mpsstr[i][j], '__iter__'):
				# peps[(i,j)].axis([1,2,3,4],[0,0,0,0]).assign(astensor(mpsstr[i][j]))
				peps[(i,j)][:,0,0,0,0] = astensor(mpsstr[i][j])
			else:
				peps[(i, j)][mpsstr[i][j], 0, 0, 0, 0] = 1.
	return peps
