

import warnings

from numpy import zeros, inner, log, asarray, power
from scipy.linalg import svd, qr
from scipy.linalg import norm as dnorm
# from numpy.linalg.linalg import LinAlgError

__all__ = ['svd2', 'qr2', 'moveSelectedIndexForward', 'moveSelectedIndexBackward', 'prodTuple',
'matrixDeparallelisation', 'svdTruncate', 'measure_entropy_dense', 'measure_renyi_entropy_dense']

# the desired svd
def svd2(A):
	"""
	stable singular value decomposition.
	gesdd is faster but sometimes do not converge,
	gesvd always converges in my experience.
	therefore, try using gesdd first, if not converge,
	then use gesvd instead.
	"""
	# return svd(A, overwrite_a=True, full_matrices=False, lapack_driver='gesvd')
	try:
		return svd(A, overwrite_a=False, full_matrices=False, lapack_driver='gesdd')
	except:
		return svd(A, overwrite_a=True, full_matrices=False, lapack_driver='gesvd')

def qr2(a):
	"""
	QR decomposition
	"""
	return qr(a, overwrite_a=True, mode='economic', pivoting=False)

# def svd2s(a, k=6, v0=None):
# 	"""
# 	Iterative svd, implemented using arpack package. it will
# 	be fast if we only want to get a small portion of the
# 	singular values, say 200 out of 1000.
# 	I am not sure the output order is descending or not
# 	so the code below just make sure the singular values are
# 	in descending order
# 	"""
# 	u, s, v = svds(a, k=k, v0=v0)
# 	I = s.argsort()[::-1]
# 	return u[:, I], s[I], v[I, :]

def moveSelectedIndexForward(a, I):
	na = len(a)
	nI = len(I)
	b = [None]*na
	k1 = 0
	k2 = nI
	for i in range(na):
		s = 0
		while s != nI:
			if i == I[s]:
				b[s] = a[k1]
				k1 += 1
				break
			s += 1
		if s == nI:
			b[k2] = a[k1]
			k1 += 1
			k2 += 1
	return type(a)(b)

def moveSelectedIndexBackward(a, I):
	na = len(a)
	nI = len(I)
	nr = na - nI
	b = [None]*na
	k1 = 0
	k2 = 0
	for i in range(na):
		s = 0
		while s != nI:
			if i == I[s]:
				b[nr + s] = a[k1]
				k1 += 1
				break
			s += 1
		if s == nI:
			b[k2] = a[k1]
			k2 += 1
			k1 += 1
	return type(a)(b)

def prodTuple(l):
	# assert(len(l) > 0)
	s = 1
	for i in l:
		s *= i
	return s

# deparallelisation
# this function assume not all the elements of cola are zeros
def isTwoColumnParallel(cola, colb, tol=1.0e-12):
	assert(len(cola) == len(colb))
	n = len(cola)
	assert(n > 0)
	colanonzeros = []
	for i in range(n):
		if abs(cola[i])>tol:
			colanonzeros.append(i)
	assert(len(colanonzeros) > 0)
	factor = colb[colanonzeros[0]]/cola[colanonzeros[0]]
	diff = colb - factor*cola
	for i in diff:
		if abs(i) > tol:
			return False, None
	return True, factor

# clear the column of m that are all zeros and return the index
def getRidOfZeroCol(m, tol=1.0e-12, verbose=False):
	assert(m.ndim==2)
	s1 = m.shape[0]
	s2 = m.shape[1]
	zerocols = []
	for j in range(s2):
		allzero = True
		for i in range(s1):
			if (abs(m[i,j]) > tol):
				allzero=False
				break
		if (allzero == True):
			if verbose:
				print('all elements of column ', j, ' are zero.')
			zerocols.append(j)
	ns = s2 - len(zerocols)
	if (ns == 0 and verbose == True):
		print('all the columns are zero.')
	mout = zeros((s1, ns), dtype=m.dtype)
	j = 0
	for i in range(s2):
		if i not in zerocols:
			mout[:, j] = m[:, i]
			j += 1
	return mout, zerocols

def matrixDeparallelisationNoZeroCols(m, tol=1.0e-12, verbose=False):
	s1 = m.shape[0]
	s2 = m.shape[1]
	K = []
	T = zeros((s2, s2), dtype=m.dtype)
	for j in range(s2):
		exist = False
		for i in range(len(K)):
			p, factor = isTwoColumnParallel(K[i], m[:, j], tol)
			if p==True:
				if verbose:
					print('column ', i, 'is in parallel with column ', j)
				T[i, j] = factor
				exist = True
				break
		if not exist:
			K.append(m[:, j])
			nK = len(K)
			T[nK-1, j] = 1
	nK = len(K)
	M = zeros((s1, nK), dtype=m.dtype)
	for j in range(nK):
		M[:, j] = K[j]
	return M, T[:nK, :]

def matrixDeparallelisation(m, tol=1.0e-12, verbose=False):
	# assert(m.ndim==2)
	if m.ndim != 2:
		raise ValueError('a matrix is required for matrix deparallelisation.')
	mnew, zerocols = getRidOfZeroCol(m, tol, verbose)
	M, T = matrixDeparallelisationNoZeroCols(mnew, tol, verbose)
	if (M.size == 0):
		if verbose == True:
			print('all the elements of the matrix M are 0.')
		return M, T
	Tnew = zeros((T.shape[0], m.shape[1]), dtype=T.dtype)
	j = 0
	for i in range(Tnew.shape[1]):
		if i not in zerocols:
			Tnew[:, i] = T[:, j]
			j += 1
	# assert(allclose(dot(M, Tnew), m))
	return M, Tnew

def svdTruncate(U, S, V, maxbonddimension, svdcutoff, relative_truncate, verbose):
	"""
	truncate the u, s, v resulting from svd.
	try to truncate with the threshold first,
	if the result dimension less than D, return.
	Otherwise, force truncating to D
	"""
	# assert(isinstance(U, ndarray))
	# assert(isinstance(S, ndarray))
	# assert(isinstance(V, ndarray))
	# assert(maxbonddimension > 0)
	if maxbonddimension <= 0:
		raise ValueError('maxbonddimension must be larger than 0.')
	sizem = S.size
	dim = sizem
	# assert(S.ndim == 1)
	# assert(U.shape[U.ndim-1] == S.shape[0] and S.shape[0] == V.shape[0])
	if not (S.ndim == 1 and U.shape[U.ndim-1] == S.shape[0] and S.shape[0] == V.shape[0]):
		raise ValueError('input shape mismatch for svdtruncate.')
	if (relative_truncate):
		svdcutoff = svdcutoff * dnorm(S)
	for i in range(sizem):
		if S[i] < svdcutoff:
			dim = i
			break
	if (dim == sizem and sizem <= maxbonddimension):
		if (verbose >= 2):
			print('sum:', sizem, '->', dim)
		return (U, S, V, (dim, 0.))
	if (dim > maxbonddimension):
		message = "sum: %s -> %s, maximum %s, truncation error: %s." % (
													sizem,
													dim,
													maxbonddimension,
													S[maxbonddimension])
		warnings.warn(message)
		dim = maxbonddimension
	else:
		if verbose >= 2:
			print('sum:', sizem, '->', dim)
	s = S[dim]
	return (U[..., 0:dim], S[0:dim], V[0:dim, :], (dim, s))

def measure_entropy_dense(v):
	"""
	v is a one dimensional tensor with non-negative elements,
	return the Von Neumann entropy
	"""
	v = asarray(v)
	if v.ndim != 1:
		raise ValueError('a vector is required for measure entropy.')
	a = v * v
	s = a.sum()
	a /= s
	return -inner(a, log(a))

def measure_renyi_entropy_dense(v, n):
	v = asarray(v)
	if v.ndim != 1:
		raise ValueError('a vector is required for measure entropy.')
	if n == 1:
		return measure_entropy_dense(v)
	v = v / dnorm(v)
	a = power(v, 2*n)
	return (1./(1-n))*log(a.sum())
