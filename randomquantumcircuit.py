

from rqc.gates import Xh, Yh, H, T, CZ
from rqc.core import QuantumCircuit2D, generateQState2D

from numpy import sqrt, zeros, bool_
from numpy.random import randint
import time
from copy import deepcopy

class google_circuit:
	"""
	cloumn major storage
	docstring for google_circuit
	"""
	def __init__(self, m, n):
		self.m = m
		self.n = n

		self.__initialize_configs()

		self.XYT = {0:Xh, 1:Yh, 2:T}
		# self.XYT = {0:self.op['sx'], 1:self.op['sy'], 2:diag([1, exp(1j*pi/4)])}
		self.Xoperated = zeros((m, n), dtype=bool_)
		self.Yoperated = zeros((m, n), dtype=bool_)
		self.Toperated = zeros((m, n), dtype=bool_)
		self.CZoperated = zeros((m, n), dtype=bool_)

		self.bonderrors = []

	@property
	def shape(self):
		return (self.m, self.n)

	def __initialize_configs(self):
		self.present_config = 0
		# ----this is used for testing------
		# ControlZ = expm(1j*(kron(self.op['sx'], self.op['sx']) + kron(self.op['sy'], self.op['sy'])))
		# ControlZ = ControlZ.reshape((2,2,2,2))
		# ----this is used for testing------
		ControlZ = CZ
		config1 = []
		config2 = []
		config3 = []
		config4 = []
		config5 = []
		config6 = []
		config7 = []
		config8 = []
		config1sites = []
		config2sites = []
		config3sites = []
		config4sites = []
		config5sites = []
		config6sites = []
		config7sites = []
		config8sites = []

		# 1 configuration
		for j in range(0, self.n-1, 2):
			for i in range(((j//2+1)%2), self.m, 2):
				config1.append((((i, j), (i, j+1)), ControlZ))
				config1sites.append((i, j))
				config1sites.append((i, j+1))

		# 2 configuration
		for j in range(0, self.n-1, 2):
			for i in range(((j//2)%2), self.m, 2):
				config2.append((((i, j), (i, j+1)), ControlZ))
				config2sites.append((i, j))
				config2sites.append((i, j+1))

		# 5 configuration
		for j in range(1, self.n-1, 2):
			for i in range(((j//2+1)%2), self.m, 2):
				config5.append((((i, j), (i, j+1)), ControlZ))
				config5sites.append((i, j))
				config5sites.append((i, j+1))

		# 6 configuration
		for j in range(1, self.n-1, 2):
			for i in range(((j//2)%2), self.m, 2):
				config6.append((((i, j), (i, j+1)), ControlZ))
				config6sites.append((i, j))
				config6sites.append((i, j+1))

		# 3 configuration
		for i in range(1, self.m-1, 2):
			for j in range(((i//2+1)%2), self.n, 2):
				config3.append((((i, j), (i+1, j)), ControlZ))
				config3sites.append((i, j))
				config3sites.append((i+1, j))

		# 4 configuration
		for i in range(1, self.m-1, 2):
			for j in range(((i//2)%2), self.n, 2):
				config4.append((((i, j), (i+1, j)), ControlZ))
				config4sites.append((i, j))
				config4sites.append((i+1, j))

		# 7 configuration
		for i in range(0, self.m-1, 2):
			for j in range(((i//2)%2), self.n, 2):
				config7.append((((i, j), (i+1, j)), ControlZ))
				config7sites.append((i, j))
				config7sites.append((i+1, j))

		# 8 configuration
		for i in range(0, self.m-1, 2):
			for j in range(((i//2+1)%2), self.n, 2):
				config8.append((((i, j), (i+1, j)), ControlZ))
				config8sites.append((i, j))
				config8sites.append((i+1, j))

		self.configs=[(config1, config1sites), (config2, config2sites), (config3, config3sites), \
		(config4, config4sites), (config5, config5sites), (config6, config6sites), \
		(config7, config7sites), (config8, config8sites)]

	def __generate_one_body_gates(self):
		one_body_gates = []
		Xoperated = zeros((self.m, self.n), dtype=bool_)
		Yoperated = zeros((self.m, self.n), dtype=bool_)
		for i in range(self.m):
			for j in range(self.n):
				# Place a gate at qubit q only if this qubit is
				# occupied by a CZ gate in the previous cycle
				if self.CZoperated[i, j] == True:
					# Place a T gate at qubit q if there are no single
					# qubit gates in the previous cycles at qubit q except
					# for the initial cycle of Hadamard gates
					if self.Toperated[i, j] == False:
						one_body_gates.append(((i, j), self.XYT[2]))
						self.Toperated[i, j] = True
					else:
						# Any gate at qubit q should be different from
						# the gate at qubit q in the previous cycle.
						if self.Xoperated[i, j] == True:
							one_body_gates.append(((i, j), self.XYT[1]))
							Yoperated[i, j] = True
						else:
							if self.Yoperated[i, j] == True:
								one_body_gates.append(((i, j), self.XYT[0]))
								Xoperated[i, j] = True
							else:
								r = randint(low=0, high=2)
								one_body_gates.append(((i, j), self.XYT[0]))
								if r == 0:
									Xoperated[i, j] = True
								else:
									Yoperated[i, j] = True
		self.Xoperated = Xoperated
		self.Yoperated = Yoperated
		return one_body_gates

	def generate_circuits(self, depth):
		r = QuantumCircuit2D()
		for i in range(self.m):
			for j in range(self.n):
				r.append(((i, j), H))
		for i in range(depth):
			present_config = i % 8
			config, configsites = self.configs[present_config]
			self.CZoperated.fill(False)
			for item in configsites:
				self.CZoperated[item] = True
			one_body_gates = self.__generate_one_body_gates()
			r.extend(one_body_gates)
			r.extend(config)
		for i in range(self.m):
			for j in range(self.n):
				r.append(((i, j), H))
		return r


# width
m = 8
# height
n = 8
# circuit depth
depth = 16


maxbonddimension = 100
svdcutoff = 1.0e-12

print('m', m, 'n', n, 'depth', depth)

sample = randint(low=0, high=2, size=(m, n))

circuit_generator = google_circuit(m, n)

circuit2d = circuit_generator.generate_circuits(depth)

peps = generateQState2D([[0]*n]*m)
print('number of gate', len(circuit2d))
print('evolving the peps...')
circuit2d.apply(peps, maxbonddimension, svdcutoff, verbose=1)
print('bond dimension of the resulting peps...')
print(peps.bond_dimensions())


b = generateQState2D(sample)
s = b.cross(peps, scale_factor=sqrt(2.))
s = abs(s)
print('the 2d probability', s*s)
