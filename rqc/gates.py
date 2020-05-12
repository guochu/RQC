


# -*- coding: utf-8 -*-
# @Author: guochu
# @Date:   2018-08-26 15:17:23
# @Last Modified by:   guochu
# @Last Modified time: 2019-01-05 11:44:29
from numpy import diag, pi, exp, sin, cos, sqrt, kron

from rqc.tensor import eye, astensor

# one body gates
X = astensor([[0., 1.], [1., 0.]])
Y = astensor([[0., -1j], [1j, 0.]])
Z = astensor([[1., 0.], [0., -1.]])
S = astensor([[1., 0.], [0., 1j]])
_I = eye(2)
H = (X + Z) / sqrt(2)

_UP = astensor([[1., 0.], [0., 0.]])
_DOWN = astensor([[0., 0.], [0., 1.]])

def R(k):
	return astensor([[1., 0.], [0., exp(pi*1j/(2.**(k-1)))]])


def ROTATION(theta):
	theta = 2.*pi*theta
	return astensor([[cos(theta), sin(theta)], [sin(theta), -cos(theta)]])

def Rx(theta):
	theta = pi*theta/2
	return astensor([[cos(theta), -1j*sin(theta)], [-1j*sin(theta), cos(theta)]])

def Ry(theta):
	theta = pi*theta/2
	return astensor([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def Rz(theta):
	theta = 2.*pi*theta
	return astensor([[1., 0.], [0., exp(1j*theta)]])

Xh = astensor([[1+1j, 1-1j], [1-1j, 1+1j]])/2
Yh = astensor([[1j, -1j], [1j, 1j]])/sqrt(2*1j)
T = astensor(diag([1, exp(1j*pi/4)]))

# two body gates
CZ = astensor(diag([1.,1.,1.,-1.]))
CNOT = astensor([[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,0.,1.], [0.,0.,1.,0.]])
SWAP = astensor([[1.,0.,0.,0.], [0.,0.,1.,0.], [0.,1.,0.,0.], [0.,0.,0.,1.]])
ISWAP = astensor([[1.,0.,0.,0.], [0.,0.,1j,0.], [0.,1j,0.,0.], [0.,0.,0.,1.]])

def CONTROL(u):
	# assert(u.shape==(2,2))
	u = astensor(u)
	assert(u.shape[0] == u.shape[1])
	return kron(_UP, eye(u.shape[0])) + kron(_DOWN, u)

# three body gates
def CONTROLCONTROL(u):
	u = astensor(u)
	assert(u.shape[0] == u.shape[1])
	# assert(u.shape==(2,2))
	Iu = eye(u.shape[0])
	return kron(_UP, kron(_UP, Iu)) + kron(_UP, kron(_DOWN, Iu)) + kron(_DOWN, kron(_UP, Iu)) + kron(_DOWN, kron(_DOWN, u))

TOFFOLI = CONTROLCONTROL(X)
