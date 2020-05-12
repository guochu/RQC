

from numpy import ndarray, asarray
from numpy import diag as ddiag
from numpy import inner, kron, tensordot
from numpy import zeros as npzeros
from numpy import ones as npones
from numpy import eye as npeye
from numpy import empty as npempty
# from numpy.random import rand as nprand
from scipy.linalg import norm as dnorm
from numpy import result_type, sqrt
from .util import *
from scipy.linalg import expm as dexpm

__all__ = ['istensor', 'astensor', 'tensor', 'ones', 'zeros', 'eye', 'empty', 'result_type']

def istensor(m):
	return isinstance(m, (ndarray, tensor))

def group_extent(extent, idx):
	if not hasattr(idx, '__iter__'):
		idx = (idx,)
	n = len(idx)
	ext = [None]*(n)
	l = 0
	for i in range(n):
		s = 1
		for j in range(l, l + idx[i]):
			s = s * extent[j]
		ext[i] = s
		l += idx[i]
	return ext

def svdDecompose(a, axes):
	"""
	the tensor index specified by axes will be moved to the end
	a new tensor is obtained by transposing the original one
	according to this new index sequence
	the output u, s, v will arrange the index according to
	the new tensor
	"""
	# assert(a.size > 0)
	if a.size==0:
		raise ValueError('the input tensor must not be empty for svd.')
	n = a.rank
	nI = len(axes)
	dim = [i for i in range(n)]
	dim = moveSelectedIndexBackward(dim, axes)
	b = a.transpose(dim)
	s1 = s2 = 1
	# print(b.shape)
	ushape = [b.shape[i] for i in range(n-nI)]
	vshape = [b.shape[i] for i in range(n-nI, n)]
	for i in range(n-nI):
		s1 *= b.shape[i]
	for i in range(n-nI, n):
		s2 *= b.shape[i]
	u, s, v = svd2(b.reshape((s1, s2)))
	md = len(s)
	ushape = ushape + [md]
	vshape = [md] + vshape
	u = u.reshape(ushape)
	v = v.reshape(vshape)
	return u, s, v

def astensor(buf, dtype=None, order='C'):
	"""
	"""
	return tensor(asarray(buf, dtype=dtype, order=order))

class tensor(ndarray):
	"""docstring for tensor"""
	def __new__(cls, input_array):
		return asarray(input_array).view(cls)

	@property
	def rank(self):
		return self.ndim

	def tie(self, axes):
		return self.reshape(group_extent(self.shape, axes))

	def diag(self):
		return type(self)(ddiag(self))

	def cross(self, other, conj=True):
		if (self.shape != other.shape):
			raise ValueError('shape mis-match for cross.')
		if conj==True:
			return inner(self.reshape((self.size,)).conj(), other.reshape((other.size,)))
		else:
			return inner(self.reshape((self.size,)), other.reshape((other.size,)))

	def norm(self):
		return dnorm(self.reshape((self.size)))

	def contract(self, b, axes):
		return type(self)(tensordot(self, b, axes))

	def directSum(self, b, axes):
		if not self.size:
			return b.copy()
		if b is None:
			return self.copy()
		# assert(a.dtype==b.dtype)
		# assert(self.rank == b.rank)
		if self.rank != b.rank:
			raise ValueError('directsum requires two tensors of same rank.')
		dimc = [None]*(self.rank)
		dim = [0]*(self.rank)
		for i in range(self.rank):
			if i in axes:
				dim[i] = self.shape[i]
				dimc[i] = self.shape[i] + b.shape[i]
			else:
				dimc[i] = self.shape[i]
		c = type(self)(zeros(dimc, dtype = result_type(self.dtype, b.dtype)))
		r = [slice(0, self.shape[i]) for i in range(self.rank)]
		c[tuple(r)] = self
		for i in range(self.rank):
			r[i] = slice(dim[i], dimc[i])
		c[tuple(r)] = b
		return c

	def fusion(self, b, axes):
		assert(self.size)
		if self.size == 0:
			raise ValueError('fusion require a non empty tensor.')
		# assert(len(axes)==2)
		# assert(len(axes[0]) == len(axes[1]))
		if not (len(axes)==2 and len(axes[0]) == len(axes[1])):
			raise ValueError('wrong input axes for fusion.')
		nI = len(axes[0])
		a1 = None
		b1 = None
		ranka = self.rank
		# assert(nI <= ranka)
		if nI > ranka:
			raise IndexError('index of a out of range.')
		indexa = [i for i in range(ranka)]
		indexa = moveSelectedIndexBackward(indexa, axes[0])
		a1 = self.transpose(indexa)
		sizem = prodTuple(a1.shape[(ranka-nI):])
		a1 = a1.reshape(a1.shape[:(ranka-nI)] + (sizem,))
		if b is not None:
			rankb = b.rank
			# assert(nI <= rankb)
			if nI > rankb:
				raise IndexError('index of b out of range.')
			indexb = [i for i in range(rankb)]
			indexb = moveSelectedIndexForward(indexb, axes[1])
			b1 = b.transpose(indexb)
			sizem = prodTuple(b1.shape[:nI])
			b1 = b1.reshape((sizem,)+b1.shape[nI:])
		return a1, b1

	def svd(self, axes, maxbonddimension=-1, svdcutoff=1.0e-10, verbose=0):
		bonderror = (0, 0.)
		u, s, v = svdDecompose(self, axes)
		if maxbonddimension > 0:
			u, s, v, bonderror = svdTruncate(u, s, v, maxbonddimension, svdcutoff, 1, verbose)
		return type(self)(u), type(self)(s), type(self)(v), bonderror

	def deparallelise(self, axes, tol=1.0e-12, verbose=0):
		n = self.rank
		nI = len(axes)
		dim = [i for i in range(n)]
		dim = moveSelectedIndexBackward(dim, axes)
		b = self.transpose(dim)
		s1 = s2 = 1
		ushape = [b.shape[i] for i in range(n-nI)]
		vshape = [b.shape[i] for i in range(n-nI, n)]
		for i in range(n-nI):
			s1 *= b.shape[i]
		for i in range(n-nI, n):
			s2 *= b.shape[i]
		b = b.reshape((s1, s2))
		u, v = matrixDeparallelisation(b, tol, verbose)
		# assert(u.shape[1] == v.shape[0])
		if u.shape[1] != v.shape[0]:
			raise Exception('unknown error in deparallelise.')
		md = u.shape[1]
		ushape = ushape + [md]
		vshape = [md] + vshape
		u = u.reshape(ushape)
		v = v.reshape(vshape)
		return type(self)(u), type(self)(v)

	def qr(self, axes):
		"""
		QR decomposition
		"""
		s1 = 1
		s2 = 1
		N1 = len(axes)
		newindex = [i for i in range(self.rank)]
		dimu=[0]*(self.rank-N1+1)
		dimv=[0]*(N1+1)
		newindex = moveSelectedIndexBackward(newindex, axes)
		a1 = self.transpose(newindex)
		for i in range(self.rank-N1):
			dimu[i] = a1.shape[i]
			s1 *= a1.shape[i]
		for i in range(self.rank-N1, self.rank):
			dimv[i-self.rank+N1+1] = a1.shape[i]
			s2 *= a1.shape[i]
		u, v = qr2(a1.reshape(s1, s2))
		s = v.shape[0]
		dimu[-1] = s
		dimv[0] = s
		return type(self)(u.reshape(dimu)), type(self)(v.reshape(dimv))

	def expm(self, axes=(1,)):
		N1 = len(axes)
		perm = [i for i in range(self.rank)]
		perm = moveSelectedIndexBackward(perm, axes)
		a = self.transpose(perm)
		m = prodTuple(a.shape[:(self.rank-N1)])
		n = prodTuple(a.shape[(self.rank-N1):])
		if m != n:
			raise ValueError('square matrix is required.')
		t2 = dexpm(a.reshape((m, n)))
		return type(self)(t2.reshape(a.shape))

	def purge(self, tol=1.0e-12):
		self[abs(self) < tol] = 0

	def entropy(self):
		return self.renyi_entropy(1)

	def renyi_entropy(self, n=2):
		return measure_renyi_entropy_dense(self, n)

	def kron(self, other):
		if not isinstance(other, ndarray):
			raise TypeError('kron require two tensors.')
		return kron(self, other)

	def sqrt(self):
		return sqrt(self)

	def __repr__(self):
		return repr(asarray(self))

def ones(shape, dtype=float, order='C'):
	return tensor(npones(shape, dtype=dtype, order=order))

def zeros(shape, dtype=float, order='C'):
	return tensor(npzeros(shape, dtype=dtype, order=order))

def eye(d, dtype=float, order='C'):
	return tensor(npeye(d, dtype=dtype, order=order))

def empty(shape, dtype=float, order='C'):
	return tensor(npempty(shape, dtype=dtype, order=order))
