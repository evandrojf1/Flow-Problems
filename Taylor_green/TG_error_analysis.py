'''
TAYLOR GREEN VORTEX ERROR ANALYSIS!
'''

from fenics import *
from chorin import chorin_method
from ipcs import ipcs_method
from galerkin import galerkin_method
import numpy as np
import matplotlib.pyplot as plt
import time

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

class PeriodicDomain(SubDomain):
	def inside(self, x, on_boundary):
		return bool((near(x[0], -1) or near(x[1], -1)) and (not ((near(x[0], -1) and near(x[1], 1)) or (near(x[0], 1) and near(x[1], -1)))) and on_boundary)

	def map(self, x, y):
		if near(x[0], 1) and near(x[1], 1):
			y[0] = x[0] - 2.
			y[1] = x[1] - 2.
		elif near(x[0], 1):
			y[0] = x[0] - 2.
			y[1] = x[1]
		else:   # near, 1)
			y[0] = x[0]
			y[1] = x[1] - 2.

c_d = PeriodicDomain()

def taylor_green(n):
	mesh = UnitSquareMesh(n, n)
	nu = 1/100
	
	U_e = ('-(cos(pi*(x[0]))*sin(pi*(x[1]))) * exp(-2.0*nu*pi*pi*t)',
		   ' (cos(pi*(x[1]))*sin(pi*(x[0]))) * exp(-2.0*nu*pi*pi*t)')


	P_e = '-0.25*(cos(2*pi*(x[0])) + cos(2*pi*(x[1]))) * exp(-4.0*nu*pi*pi*t)'

	u_e = Expression(U_e, degree=3, t=0, nu=float(nu))

	K_e = 0.5*assemble(inner(u_e, u_e)*dx(mesh))
	
	###############CHORIN###############
	(u1,p1, time1) = chorin_method(mesh, c_d, U_e, P_e)

	K1 = 0.5*assemble(inner(u1, u1)*dx(mesh))
	er1 = K_e - K1

	
	###############IPCS###############
	(u2,p2,time2) = ipcs_method(mesh, c_d, U_e, P_e)
	K2 = 0.5*assemble(inner(u2, u2)*dx(mesh))
	er2 = K_e - K2
	
	###############GALERKIN###############
	(u3, p3, time3) = galerkin_method(mesh, c_d, U_e, P_e)
	K3 = 0.5*assemble(inner(u3, u3)*dx(mesh))
	er3 = K_e - K3

	degrees1 = u1.vector().size() + p1.vector().size()
	degrees2 = u3.vector().size() + p3.vector().size()

	return mesh, er1, er2, er3, time1, time2, time3, degrees1, degrees2
	#return mesh, er3, time3


meshes = [2**i for i in range(2,7)]
error1 = []
tempo1 = []
error2 = []
tempo2 = []
error3 = []
tempo3 = []
degree1 = []
degree2 = []

for n in meshes:
	mesh, er1, er2, er3, t1, t2, t3, d1, d2 = taylor_green(n)
	#mesh, er3, t3 = taylor_green(n)
	tempo1.append(t1)
	tempo2.append(t2)
	tempo3.append(t3)
	degree1.append(d1)
	degree2.append(d2)
	error1.append(assemble(0.034*er1**2*dx(mesh)))
	error2.append(assemble(0.01*er2**2*dx(mesh)))
	error3.append(assemble(er3**2*dx(mesh)))


fig, (ax,bx) = plt.subplots(2,1)

ax.loglog(degree1, error1, color = 'Red', marker = '*', linestyle = '-', label = 'CHORIN')
ax.loglog(degree1, error2, color = 'Blue', marker = 'o', linestyle = '--', label = 'IPCS')
ax.loglog(degree2, error3, color = 'Green', marker = 's', linestyle = '-.', label = 'GALERKIN')
ax.set_ylabel('error')

bx.loglog(degree1, tempo1, color = 'Red', marker = '*', linestyle = '-')
bx.loglog(degree1, tempo2, color = 'Blue', marker = 'o', linestyle = '--')
bx.loglog(degree2, tempo3, color = 'Green', marker = 's', linestyle = '-.')
bx.set_xlabel('mesh')
bx.set_ylabel('time [s]')

ax.legend(loc='best')
fig.suptitle('Taylor Green Vortex')
fig.savefig('results.png')


