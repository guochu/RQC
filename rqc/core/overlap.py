
from numpy import sqrt

__all__ = ['rescaled_overlap', 'close_peps']

def tensor2scalar(m):
	assert(m.size==1)
	m1 = m.reshape((m.size,))
	return m1[0]

def close_peps(pepsA, pepsB, op={}):
	m, n = pepsA.shape
	# assert(n > 0 and m > 0)
	# assert(m == pepsB.shape[0] and n == pepsB.shape[1])
	if not (n > 0 and m > 0 and m == pepsB.shape[0] and n == pepsB.shape[1]):
		raise ValueError('shape mismatch for close 2d state.')
	ms = [None]*m
	for i in range(m):
		ms[i] = [None]*n
		for j in range(n):
			tmp = pepsB[(i, j)]
			t = op.get((i, j))
			if t is not None:
				tmp = t.contract(tmp, ((1,), (0,)))
			ms[i][j] = pepsA[(i, j)].conj().contract(tmp, ((0,), (0,)))
	for i in range(m):
		ms[i][0], tmp = ms[i][0].fusion(None, ((1,5),(0,0)))
		ms[i][0] = ms[i][0].transpose((6,0,1,2,3,4,5))
		for j in range(n-1):
			ms[i][j], ms[i][j+1] = ms[i][j].fusion(ms[i][j+1], ((2,5),(1,5)))
		ms[i][n-1], tmp = ms[i][n-1].fusion(tmp, ((2,5),(0,0)))
	for j in range(n):
		# assert(ms[0][j].shape[1]==1)
		# assert(ms[0][j].shape[3]==1)
		if not (ms[0][j].shape[1]==1 and ms[0][j].shape[3]==1):
			raise Exception('shape error for close peps.')
		ms[0][j], tmp = ms[0][j].fusion(None, ((1,3), (0,0)))
		ms[0][j] = ms[0][j].transpose((4,0,1,2,3))
		for i in range(m-1):
			ms[i][j], ms[i+1][j] = ms[i][j].fusion(ms[i+1][j], ((2,3), (1,3)))
		ms[m-1][j], tmp = ms[m-1][j].fusion(None, ((2,3), (0,0)))
	return ms

def brute_force_update(r, m):
	L = len(m)
	# assert(r.rank == L+2)
	if r.rank != L+2:
		raise Exception('wrong input tensor rank.')
	s = m[-1].contract(r, ((0,2), (L,L+1)))
	for i in range(L-2, 0, -1):
		s = m[i].contract(s, ((0,2), (L+1, 0)))
	s = m[0].contract(s, ((1,0,2), (L, L+1, 0)))
	new_shape = (1,) + s.shape + (1,)
	return s.reshape(new_shape)

def brute_force(ms):
	m = len(ms)
	# tmp = ms[2][2]
	# print(tmp.shape)
	# u, s, v, bet = tmp.svd((1,3), 10000, 1.0e-10, 2)
	# assert(False)
	if m<=1:
		raise ValueError('the size of PEPS is less than 1.')
	# assert(m > 1)
	# contract the first row
	n = len(ms[0])
	# assert(n > 1)
	if n <= 1:
		raise ValueError('the size of PEPS is less than 1.')
	mpsu = ms[0]
	for i in range(len(mpsu)):
		# assert(mpsu[i].shape[0]==1)
		if mpsu[i].shape[0]!=1:
			raise ValueError('shape error for brute force overlap.')
		mpsu[i], res = mpsu[i].fusion(None, ((0,3), (0,0)))
		mpsu[i] = mpsu[i].transpose((0,2,1))
	r = mpsu[0].contract(mpsu[1], ((2,), (0,)))
	for i in range(2, n):
		r = r.contract(mpsu[i], ((i+1,), (0,)))
	# r is n+2 dimensional tensor, with the first and last dimension to be 1
	for i in range(1, m-1):
		r = brute_force_update(r, ms[i])
	mpsd = ms[-1]
	for i in range(len(mpsd)):
		mpsd[i], res = mpsd[i].fusion(None, ((0,3),(0,0)))
		mpsd[i] = mpsd[i].transpose((0,2,1))
	r = mpsd[-1].contract(r, ((1,2), (n, n+1)))
	for i in range(n-2, 0, -1):
		r = mpsd[i].contract(r, ((1,2), (i+2, 0)))
	return mpsd[0].contract(r, ((0,1,2), (1,2,0))), 0.

def prepare_ms(ms):
	"""
	shape of output ms
	  1 --  0     2 ------------- 0
	0	       1					 1

	0		   0                  0
	  1 --  1     2 -------------	 1
	2		   3				  2
	----------------------------------
	0		   1					 1
	  1 --  0	  2 ------------- 0
	"""
	m = len(ms)
	# assert(m>1)
	if m <= 1:
		raise ValueError('input should size larger than 1.')
	n = len(ms[0])
	mout = ms
	# for i in range(len(ms)):
	# 	mout[i] = [None]*len(ms[i])
	for i in range(1, len(ms)-1):
		# assert(ms[i][0].shape[1]==1)
		if ms[i][0].shape[1]!=1:
			raise ValueError('input shape error.')
		tmp = ms[i][0]
		mout[i][0] = tmp.reshape((tmp.shape[0], tmp.shape[2], tmp.shape[3]))
		# assert(ms[i][n-1].shape[2]==1)
		if ms[i][n-1].shape[2]!=1:
			raise ValueError('input shape error.')
		tmp = ms[i][n-1]
		mout[i][n-1] = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[3]))
	for j in range(1,n-1):
		# assert(ms[0][j].shape[0]==1)
		if ms[0][j].shape[0]!=1:
			raise ValueError('input shape error.')
		tmp = ms[0][j]
		tmp = tmp.reshape((tmp.shape[1], tmp.shape[2], tmp.shape[3]))
		mout[0][j] = tmp.transpose((0,2,1))
		tmp = ms[m-1][j]
		# assert(tmp.shape[3]==1)
		if tmp.shape[3]!=1:
			raise ValueError('input shape error.')
		tmp = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2]))
		mout[m-1][j] = tmp.transpose((1,0,2))
	mout[0][0] = ms[0][0].reshape((ms[0][0].shape[2], ms[0][0].shape[3])).transpose((1,0))
	mout[0][n-1] = ms[0][n-1].reshape((ms[0][n-1].shape[1], ms[0][n-1].shape[3]))
	mout[m-1][0] = ms[m-1][0].reshape((ms[m-1][0].shape[0], ms[m-1][0].shape[2]))
	mout[m-1][n-1] = ms[m-1][n-1].reshape((ms[m-1][n-1].shape[0], ms[m-1][n-1].shape[1])).transpose((1,0))
	return mout

def brute_force_update_1(r, m):
	L = len(m)
	# assert(r.rank == L)
	if r.rank != L:
		raise ValueError('wrong input tensor rank.')
	s = m[-1].contract(r, ((0,), (L-1,)))
	for i in range(L-2, 0, -1):
		s = m[i].contract(s, ((0,2), (L, 0)))
	return m[0].contract(s, ((0,1), (L, 0)))

def brute_force_1(ms):
	ms = prepare_ms(ms)
	m = len(ms)
	# tmp = ms[2][2]
	# print(tmp.shape)
	# u, s, v, bet = tmp.svd((1,3), 10000, 1.0e-10, 2)
	# assert(False)
	if m<=1:
		raise ValueError('the size of PEPS is less than 1.')
	assert(m > 1)
	# contract the first row
	n = len(ms[0])
	assert(n > 1)
	mpsu = ms[0]
	r = mpsu[0]
	for i in range(1, n):
		r = r.contract(mpsu[i], ((r.rank-1,), (0,)))

	# r is n+2 dimensional tensor, with the first and last dimension to be 1
	for i in range(1, m-1):
		r = brute_force_update_1(r, ms[i])
	mpsd = ms[-1]
	r = r.contract(mpsd[-1], ((r.rank-1,), (1,)))
	for i in range(n-2, 0, -1):
		r = r.contract(mpsd[i], ((r.rank-2, r.rank-1), (1,2)))
	return tensor2scalar(r.contract(mpsd[0], ((0,1), (0,1))))


def rescaled_overlap(pepsA, pepsB, scale_factor):
	"""
	rescale each site by scale_factor
	"""
	ms = close_peps(pepsA, pepsB)
	for i in range(len(ms)):
		for j in range(len(ms[i])):
			ms[i][j] *= scale_factor
	return brute_force_1(ms)
