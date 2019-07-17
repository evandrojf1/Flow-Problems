'''
DRIVEN CAVITY ERROR ANALYSIS!
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

def driven_cavity(n):
	mesh = UnitSquareMesh(n, n)
	h = mesh.hmin()
	V = VectorFunctionSpace(mesh, 'P', 2)
	Q = FunctionSpace(mesh, 'P', 1)

	# Define Boundaries and Boundary Condictions
	upperedge = 'near(x[1], 1)'
	noslip   = 'near(x[1], 0) || near(x[0], 1) || near(x[0], 0)'

	bcu_noslip  = DirichletBC(V, Constant((0.0, 0.0)), noslip)
	bcu_top  = DirichletBC(V, Constant((1.0,0.0)), upperedge)
	bcu = [bcu_noslip, bcu_top]
	bcp = []

	# Define variacional parameters
	u = TrialFunction(V)
	p = TrialFunction(Q)
	v = TestFunction(V)
	q = TestFunction(Q)

	# Set parameter values
	T = 2.5           # final time
	dt = 0.2*h/1. # time step CFL with 1 = max. velocity

	nu = 1/1000             # kinematic viscosity
	k = dt
	u0 = Constant((0.0, 0.0))
	p0 = Constant(0.0)
	f = Constant((0.0, 0.0))
	rho = 1
	n = FacetNormal(mesh)

	teste = 0
	psi_e = -0.0585236
	###############CHORIN###############
	inicio1 = time.time()
	(u1,p1) = chorin_method(k, u, v, nu, f, p, q, bcu, bcp, dt, V, Q, teste, u0, p0, T)
	fim1 = time.time()
	time1 = (fim1 - inicio1)

	psi1 = stream_function(u1)
	stream_min1 = np.array(psi1.vector()).min()
	er1 = psi_e - stream_min1


	###############IPCS###############
	inicio2 = time.time()
	(u2,p2) = ipcs_method(rho, n, k, u, v, nu, f, p, q, bcu, bcp, dt, V, Q, teste, u0, p0, T)
	fim2 = time.time()
	time2 = (fim2 - inicio2)
	
	psi2 = stream_function(u2)
	stream_min2 = np.array(psi2.vector()).min()
	er2 = psi_e - stream_min2

	###############IPCS###############
	(u3, p3, time3) = galerkin_method(mesh)
	psi3 = stream_function(u3)
	stream_min3 = np.array(psi3.vector()).min()
	er3 = psi_e - stream_min3

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
	mesh, er1, er2, er3, t1, t2, t3, d1, d2 = driven_cavity(n)
	tempo1.append(t1)
	tempo2.append(t2)
	tempo3.append(t3)
	degree1.append(d1)
	degree2.append(d2)
	error1.append(assemble(0.02*er1**2*dx(mesh)))
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
fig.suptitle('Driven Cavity')
fig.savefig('Driven_cavity.png')


