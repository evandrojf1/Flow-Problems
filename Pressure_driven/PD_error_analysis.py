'''
PRESSURE DRIVEN ERROR ANALYSIS!
'''

from fenics import *
from chorin import chorin_method
from stream_function import stream_function
from ipcs import ipcs_method
from galerkin import galerkin_method
import numpy as np
import matplotlib.pyplot as plt
import time

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def pressure_driven(n):
	mesh = UnitSquareMesh(n, n)
	h = mesh.hmin()
	V = VectorFunctionSpace(mesh, 'P', 2)
	Q = FunctionSpace(mesh, 'P', 1)

	# Define Boundaries and Boundary Condictions
	inflow = 'near(x[0], 0)'
	outflow = 'near(x[0], 1)'
	walls   = 'near(x[1], 0) || near(x[1], 1)'

	bcu_noslip  = DirichletBC(V, Constant((0.0, 0.0)), walls)
	bcp_in  = DirichletBC(Q, Constant(1.0), inflow)
	bcp_out = DirichletBC(Q, Constant(0.0), outflow)
	bcu = [bcu_noslip]
	bcp = [bcp_in, bcp_out]

	# Define variacional parameters
	u = TrialFunction(V)
	p = TrialFunction(Q)
	v = TestFunction(V)
	q = TestFunction(Q)

	# Set parameter values
	T = 0.5           # final time
	dt = 0.2*h/1. # time step CFL with 1 = max. velocity

	# Set parameter values
	T = 0.5           # final time
	dt = 0.2*h/1. # time step CFL with 1 = max. velocity

	nu = 1/8             # kinematic viscosity
	rho = 1
	k = dt
	u0 = Constant((0.0, 0.0))
	p0 = Constant(0.0)
	f = Constant((0.0, 0.0))

	teste = 0
	u_e = 0.44321183655
	###############CHORIN###############
	inicio1 = time.time()
	(u1,p1) = chorin_method(k, u, v, nu, f, p, q, bcu, bcp, dt, V, Q, teste, u0, p0, T)
	fim1 = time.time()
	time1 = (fim1 - inicio1)

	u1_a = u1(1.0,0.5)[0]
	er1 = u_e - u1_a


	###############IPCS###############
	(u2,p2,time2) = ipcs_method(mesh)
	u2_a = u2(1.0,0.5)[0]
	er2 = u_e - u2_a

	###############IPCS###############
	(u3, p3, time3) = galerkin_method(mesh)
	u3_a = u3(1.0,0.5)[0]
	er3 = u_e - u3_a

	degrees1 = u1.vector().size() + p1.vector().size()
	degrees2 = u3.vector().size() + p3.vector().size()

	return mesh, er1, er2, er3, time1, time2, time3, degrees1, degrees2


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
	mesh, er1, er2, er3, t1, t2, t3, d1, d2 = pressure_driven(n)
	tempo1.append(t1)
	tempo2.append(t2)
	tempo3.append(t3)
	degree1.append(d1)
	degree2.append(d2)
	error1.append(assemble(er1**2*dx(mesh)))
	error2.append(assemble(er2**2*dx(mesh)))
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
fig.suptitle('Pressure Driven')
fig.savefig('Pressure_driven.png')


