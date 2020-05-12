

from math import sqrt as ssqrt

from rqc.tensor import eye, astensor
from .peps import generateProdPEPS as _generateProdPEPS

__all__ = ['QuantumCircuit2D', 'generateQState2D', 'OneBodyGate', 'TwoBodyGate', 'asgate']


def split_bondmpo(bondmpo):
	u, s, v, bonderror = bondmpo.svd((1,3), 10000, 1.0e-12, 0)
	sm = s.sqrt().diag()
	u = u.contract(sm, ((2,), (0,)))
	v = sm.contract(v, ((1,), (0,)))
	return u, v

def combine_bondmpo(op):
	u, v = op
	m = u.contract(v, ((2,), (0,)))
	return m.transpose((0,2,1,3))


def bondEvolution(HorV, sbondmpo, mpsj1, mpsj2, maxbonddimension, svdcutoff, verbose):
	"""
	if itertiveLevel=2, then no explicit two site mps will be built. Instead
	the two site mps is modeled by a linear operator acting from both the left
	side and right side, and this is enough to compute the two site svd by
	iterative solver.
	If iterativeLevel=1, then explicit two site mps will be built, after that
	iterative solver is used by feeding this full two site mps.
	If iterativeLevel=0, then explicit two site mps will be built, and a dense
	full svd solver will be used
	"""
	mpoj1, mpoj2 = sbondmpo
	assert(mpoj1.rank==3 and mpoj2.rank==3)
	assert(mpsj1.rank==5 and mpsj2.rank==5)
	assert(HorV in ['H', 'V'])
	m1 = mpoj1.contract(mpsj1, ((1,), (0,)))
	m2 = mpoj2.contract(mpsj2, ((2,), (0,)))
	# print(mpsj1.shape, mpsj2.shape)
	# print(m1.shape, m2.shape)
	if HorV == 'H':
		# print('apply horizontal bond evolution')
		q, r = m1.qr((1,4))
		m2 = m2.contract(r, ((0,3),(1,2)))
		u, s, v, bonderror = m2.svd((0,1,2,3), maxbonddimension, svdcutoff, verbose)
		sm = s.sqrt().diag()
		u = u.dot(sm)
		q = q.contract(u, ((4,), (0,)))
		q = q.transpose((0,1,2,4,3))
		v = sm.contract(v, ((1,),(0,)))
		v = v.transpose((1,2,0,3,4))
	else:
		# print('apply vertical bond evolution')
		q, r = m1.qr((1, 5))
		m2 = m2.contract(r, ((0,2), (1,2)))
		u, s, v, bonderror = m2.svd((0,1,2,3), maxbonddimension, svdcutoff, verbose)
		sm = s.sqrt().diag()
		u = u.dot(sm)
		q = q.contract(u, ((4,),(0,)))
		v = sm.contract(v, ((1,),(0,)))
		v = v.transpose((1,0,2,3,4))
	# print(q.shape, v.shape)
	return q, s, v, bonderror

def apply_one_body_gates(one_body_gates, peps):
	for key, value in one_body_gates.items():
		peps[key] = value.contract(peps[key], ((1,), (0,)))
	return peps

def isHorV(l1, l2):
	if (l1[0] == l2[0] and l1[1]+1==l2[1]):
		return 'H'
	elif (l1[0]+1==l2[0] and l1[1]==l2[1]):
		return 'V'
	else:
		raise ValueError("neither H or V!")

def apply_nn_two_body_gates(two_body_gates, peps, maxbonddimension=200, svdcutoff=1.0e-10, verbose=0):
	bonderror = (0, 0.)
	for key, value in two_body_gates.items():
		l1, l2 = key
		HorV = isHorV(l1, l2)
		peps[l1], s, peps[l2], bet = bondEvolution(HorV, value, \
			peps[l1], peps[l2], maxbonddimension, svdcutoff, verbose)
		bonderror = (max(bonderror[0], bet[0]), max(bonderror[1], bet[1]))
	return bonderror


class OneBodyGate:
	"""docstring for OneBodyGate"""
	def __init__(self, key, op):
		# assert(len(key)==2)
		if len(key) != 2:
			raise ValueError('wrong position for one body gate.')
		for s in key:
			if not isinstance(s, int):
				raise TypeError('position must be integer type.')
		self.key = tuple(key)
		if op.rank != 2:
			raise ValueError('one body operator should have rank 2.')
		self.op = op

	def apply(self, state, maxbonddimension=200, svdcutoff=1.0e-12, verbose=1):
		state[self.key] = self.op.contract(state[self.key], ((1,), (0,)))

	def apply_and_collect(self, state, result, maxbonddimension=200, svdcutoff=1.0e-12, verbose=1):
		return self.apply(state, maxbonddimension, svdcutoff, verbose=verbose)

	@property
	def T(self):
		return type(self)(self.key, self.op.T)

	@property
	def H(self):
		return type(self)(self.key, self.op.T.conj())

	def conj(self):
		return type(self)(self.key, self.op.conj())

	def __iter__(self):
		return iter((self.key, self.op))

	def __repr__(self):
		return repr((self.key, self.op))


class TwoBodyGate:
	"""docstring for TwoBodyGate"""
	def __init__(self, key, op):
		self.key, self.value = self.__get_normal_order(key, op)
		self.HorV = isHorV(self.key[0], self.key[1])

	def __get_normal_order(self, key, op):
		l1, l2 = key
		# assert(l1 != l2)
		if l1 == l2:
			print('wrong position for two body gate.')
		if isinstance(op, tuple):
			# assert(len(op)==2 and op[0].rank==3 and op[1].rank==3)
			# assert((l1[0] == l2[0] and l1[1]+1==l2[1]) or (l1[0]+1==l2[0] and l1[1]==l2[1]))
			if not (len(op)==2 and op[0].rank==3 and op[1].rank==3):
				raise ValueError('wrong input op for two body gate.')
			if not ((l1[0] == l2[0] and l1[1]+1==l2[1]) or (l1[0]+1==l2[0] and l1[1]==l2[1])):
				raise ValueError('wrong position for two body gate.')
		else:
			op = astensor(op)
			# assert(len(l1)==2 and len(l2)==2)
			if not (len(l1)==2 and len(l2)==2):
				raise ValueError('wrong position for two body gate.')
			if op.rank==2:
				# assert(op.shape==(4,4) or op.shape==(16,16))
				if op.shape!=(4,4):
					raise ValueError('wrong op shape for two body gate.')
				s = int(ssqrt(op.shape[0]))
				op = op.reshape((s,s,s,s))
			else:
				# assert(op.rank==4 and (op.shape==(2,2,2,2) or op.shape==(4,4,4,4)))
				if op.shape != (2,2,2,2):
					raise ValueError('wrong op shape for two body gate.')
			if ((l1[0] == l2[0] and l1[1]==l2[1]+1) or (l1[0]==l2[0]+1 and l1[1]==l2[1])):
				op = op.transpose((1,0,3,2))
				key = (l2, l1)
			op = split_bondmpo(op)
		return key, op

	def apply(self, peps, maxbonddimension=200, svdcutoff=1.0e-12, verbose=1):
		"""
		if itertiveLevel=2, then no explicit two site mps will be built. Instead
		the two site mps is modeled by a linear operator acting from both the left
		side and right side, and this is enough to compute the two site svd by
		iterative solver.
		If iterativeLevel=1, then explicit two site mps will be built, after that
		iterative solver is used by feeding this full two site mps.
		If iterativeLevel=0, then explicit two site mps will be built, and a dense
		full svd solver will be used
		"""
		maxbonddimension = 200
		l1, l2 = self.key
		mpoj1, mpoj2 = self.value
		mpsj1 = peps[l1]
		mpsj2 = peps[l2]
		assert(mpoj1.rank==3 and mpoj2.rank==3)
		m1 = mpoj1.contract(mpsj1, ((1,), (0,)))
		m2 = mpoj2.contract(mpsj2, ((2,), (0,)))
		if self.HorV == 'H':
			q, r = m1.qr((1,4))
			m2 = m2.contract(r, ((0,3),(1,2)))
			u, s, v, bonderror = m2.svd((0,1,2,3), maxbonddimension, svdcutoff, verbose=verbose)
			sm = s.sqrt().diag()
			u = u.dot(sm)
			q = q.contract(u, ((4,), (0,)))
			q = q.transpose((0,1,2,4,3))
			v = sm.contract(v, ((1,),(0,)))
			v = v.transpose((1,2,0,3,4))
		else:
			q, r = m1.qr((1, 5))
			m2 = m2.contract(r, ((0,2), (1,2)))
			u, s, v, bonderror = m2.svd((0,1,2,3), maxbonddimension, svdcutoff, verbose=verbose)
			sm = s.sqrt().diag()
			u = u.dot(sm)
			q = q.contract(u, ((4,),(0,)))
			v = sm.contract(v, ((1,),(0,)))
			v = v.transpose((1,0,2,3,4))
		peps[l1] = q
		peps[l2] = v
		# print('singular values', s)
		return [bonderror]

	def apply_and_collect(self, state, result, maxbonddimension=200, svdcutoff=1.0e-12, verbose=1):
		return self.apply(state, maxbonddimension, svdcutoff, verbose=verbose)

	@property
	def op(self):
		return combine_bondmpo(self.value)

	@property
	def T(self):
		value = self.op.transpose((2,3,0,1))
		return type(self)(self.key, split_bondmpo(value))

	@property
	def H(self):
		value = self.op.transpose((2,3,0,1)).conj()
		return type(self)(self.key, split_bondmpo(value))

	def conj(self):
		return type(self)(self.key, (self.value[0].conj(), self.value[1].conj()))

	def __iter__(self):
		return iter((self.key, self.op))

	def __repr__(self):
		return repr((self.key, self.op))

def asgate(site, op=None):
	if op is None:
		if isinstance(site, (OneBodyGate, TwoBodyGate)):
			return site
		site, op = site
	l1, l2 = site
	if isinstance(l1, int):
		return OneBodyGate(site, op)
	elif len(l1)==2:
		return TwoBodyGate(site, op)
	else:
		raise('only 1,2 body gates are supported presently.')



class QuantumCircuit2D(list):
	"""docstring for QuantumCircuit2D
	only nearest neighbour interaction
	term is allowed presently.
	"""
	def __init__(self, *args, **kwargs):
		s = list(*args, **kwargs)
		for m in s:
			self.append(m)
		for item in self:
			if not (hasattr(item, 'apply')):
				raise('item can not be applied to mps.')

	def __normalize_item(self, item):
		if not hasattr(item, 'apply'):
			item = asgate(item)
		return item

	# def __get_normal_form(self, item):
	# 	key, value = item
	# 	l1, l2 = key
	# 	if hasattr(l1, '__iter__'):
	# 		if isinstance(value, tuple):
	# 			assert(len(value)==2 and value[0].rank==3 and value[1].rank==3)
	# 			assert((l1[0] == l2[0] and l1[1]+1==l2[1]) or (l1[0]+1==l2[0] and l1[1]==l2[1]))
	# 		else:
	# 			value = astensor(value)
	# 			assert(len(l1)==2 and len(l2)==2)
	# 			if value.rank==2:
	# 				assert(value.shape==(4,4))
	# 				value = value.reshape((2,2,2,2))
	# 			else:
	# 				assert(value.rank==4 and value.shape==(2,2,2,2))
	# 			if ((l1[0] == l2[0] and l1[1]==l2[1]+1) or (l1[0]==l2[0]+1 and l1[1]==l2[1])):
	# 				value = value.transpose((1,0,3,2))
	# 				key = (l2, l1)
	# 			value = split_bondmpo(value)
	# 		return key, value
	# 	else:
	# 		value = astensor(value)
	# 		assert(value.shape[0]==2 and value.shape[1]==2)
	# 		return key, value

	# def __setitem__(self, i, item):
	# 	if hasattr(i, '__iter__'):
	# 		assert(len(i)==len(item))
	# 		super().__setitem__(i, [self.__get_normal_form(s) for s in item])
	# 	else:
	# 		super().__setitem__(i, self.__get_normal_form(item))

	def __setitem__(self, i, item):
		super().__setitem__(i, self.__normalize_item(item))

	def insert(self, i, item):
		super().insert(i, self.__normalize_item(item))

	def append(self, item):
		super().append(self.__normalize_item(item))

	def add(self, item):
		return self.append(item)

	def extend(self, items):
		super().extend([self.__normalize_item(s) for s in items])

	# def __evolve_one_gate(self, gate, peps, maxbonddimension, svdcutoff, verbose):
	# 	key, value = gate
	# 	bond = 0
	# 	error = 0.
	# 	l1, l2 = key
	# 	if hasattr(l1, '__iter__'):
	# 		HorV = isHorV(l1, l2)
	# 		peps[l1], s, peps[l2], bet = bondEvolution(HorV, value, peps[l1], \
	# 			peps[l2], maxbonddimension, svdcutoff, verbose)
	# 		bond = max(bond, bet[0])
	# 		error = max(error, bet[1])
	# 	else:
	# 		peps[key] = value.contract(peps[key], ((1,), (0,)))
	# 	return bond, error

	def apply(self, state, maxbonddimension=100, svdcutoff=1.0e-12, verbose=1):
		# self.bonderrors = []
		# for gate in self:
		# 	self.bonderrors.append(self.__evolve_one_gate(gate, peps, maxbonddimension, svdcutoff, verbose))
		bonderrors = []
		result = []
		i = 0
		for gate in self:
			r = gate.apply(state, maxbonddimension=maxbonddimension, \
				svdcutoff=svdcutoff, verbose=verbose)
			if r is not None:
				bonderrors.extend(r)
			# if i % 100 == 0:
			# 	print(mps.bond_dimensions())
			# i += 1
		return bonderrors

	def run(self, state, maxbonddimension=100, svdcutoff=1.0e-12, verbose=1):
		result = []
		bonderrors = self.apply_and_collect(state, result, maxbonddimension, svdcutoff, verbose=verbose)
		self.bonderrors = bonderrors
		return result

	def apply_and_collect(self, state, result, maxbonddimension=100, svdcutoff=1.0e-12, verbose=1):
		# self.bonderrors = []
		# for gate in self:
		# 	self.bonderrors.append(self.__evolve_one_gate(gate, peps, maxbonddimension, svdcutoff, verbose))
		bonderrors = []
		i = 0
		for gate in self:
			r = gate.apply_and_collect(state, result, maxbonddimension=maxbonddimension, \
				svdcutoff=svdcutoff, verbose=verbose)
			if r is not None:
				bonderrors.extend(r)
			# if i % 100 == 0:
			# 	print(mps.bond_dimensions())
			# i += 1
		return bonderrors

	# def expectation(self, pepsac, pepsb, scal=1., maxbonddimension=5000, svdcutoff=1.0e-12, verbose=1):
	# 	"""
	# 	<mpsa|circuit|mpsb>
	# 	"""
	# 	self.bonderrors = [None]*self.__len__()
	# 	i = 0
	# 	j = self.__len__()
	# 	pepsa = pepsac.conj()
	# 	while (i != j):
	# 		ba = pepsa.bond_dimension()
	# 		bb = pepsb.bond_dimension()
	# 		# print('i=', i, 'j=', j, 'bond dimension i=', bb, 'j=', ba)
	# 		if ba > bb:
	# 			self.bonderrors[i] =self[i].apply(pepsb, \
	# 				maxbonddimension=maxbonddimension, svdcutoff=svdcutoff, verbose=verbose)
	# 			i += 1
	# 		else:
	# 			self.bonderrors[j-1] = self[j-1].T.apply(pepsa, \
	# 				maxbonddimension=maxbonddimension, svdcutoff=svdcutoff, verbose=verbose)
	# 			j -= 1
	# 	if scal != 1.:
	# 		for i in range(len(mpsa)):
	# 			mpsa[i] *= scal
	# 	return mpsa.cross(mpsb, conj=False)


	@property
	def T(self):
		return type(self)([item.T for item in self[::-1]])

	@property
	def H(self):
		return type(self)([item.H for item in self[::-1]])

	def conj(self):
		return type(self)([item.conj() for item in self[::-1]])

	def take_in_noise(self, r1, r2):
		rr = []
		for key, value in self:
			rr.append(asgate(key, value))
			l1, l2 = key
			if hasattr(l1, '__iter__'):
				g = two_body_noise(r2)
				if g is not None:
					rr.append(self.__normalize_item((key, g)))
			else:
				g = one_body_noise(r1)
				if g is not None:
					rr.append(self.__normalize_item((key, g)))
		del self[:]
		super().extend(rr)

	def state(self, l):
		if len(l) == 0:
			raise ValueError('input shape must not be empty.')
		if isinstance(l[0], int):
			if len(l) != 2:
				raise ValueError('shape must be a tuple of 2.')
			m, n = l
			l = [[0]*n]*m
		return generateQState2D(l)


def generateQState2D(l):
	m = len(l)
	# assert(m > 0)
	if m == 0:
		raise ValueError('input must not be empty.')
	n = len(l[0])
	for i in range(1,m):
		# assert(len(l[i]) == n)
		if len(l[i]) != n:
			raise ValueError('input shoule be a square.')
	return _generateProdPEPS((m, n), [[2]*n]*m, l)
