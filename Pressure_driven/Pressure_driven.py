'''
problem: Pressure_driven channel (benchmark)

Specific parameters:
domain: UnitSquare[0,1]^2
density = 1
kinematic viscosity: 1/8
No-slip boundary conditions are imposed on top and bottom
Neumaan BC inlet and outlet. p=1 inlet, p=0 outlet
initial condition for the velocity is set to zero.

Expecting result:
The resulting flow is a vortex developing in the upper right corner and then traveling towards the center of the square as the flow evolves.

Benchmark:

functional of interest: velocity u(1,0.5) at T=0.5

'''

teste = 1

import matplotlib.pyplot as plt
from dolfin import *
from chorin import chorin_method
from ipcs import ipcs_method
import numpy as np
from mshr import *
import time

# Mesh and function spaces
n_cells = 64
mesh = UnitSquareMesh(n_cells, n_cells)
h = mesh.hmin()
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
n = FacetNormal(mesh)

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

nu = 1/8             # kinematic viscosity
rho = 1
k = dt
u0 = Constant((0.0, 0.0))
p0 = Constant(0.0)
f = Constant((0.0, 0.0))

u_e = 0.44321183655

inicio1 = time.time()
(u1,p1) = chorin_method(rho, n, k, u, v, nu, f, p, q, bcu, bcp, dt, V, Q, teste, u0, p0, T)
fim1 = time.time()

u1_a = u1(1.0,0.5)[0]

print('Tempo chorin: ', fim1-inicio1)
print('velocidade x : ', u1_a)
errorChorin = 1 - (u_e - u1_a)/u_e
print('error chorin: ', errorChorin)
print('degrees of freedom:  ', u1.vector().size() + p1.vector().size())

plot(u1)
plt.savefig('VelocityChorin'+str(i)+'.png')
plot(p1)
plt.savefig('PressureChorin'+str(i)+'.png')

inicio2 = time.time()
(u2,p2) = ipcs_method(rho, n, k, u, v, nu, f, p, q, bcu, bcp, dt, V, Q, teste, u0, p0, T)
fim2 = time.time()

u2_a = u2(1.0,0.5)[0]

print('Tempo IPCS: ', fim2-inicio2)
print('velocidade x : ', u2_a)
errorChorin = 1 - (u_e - u2_a)/u_e
print('error IPCS: ', errorChorin)
print('degrees of freedom:  ', u2.vector().size() + p2.vector().size())


plot(u2)
plt.savefig('VelocityIPCS.png')
plot(p2)
plt.savefig('PressureIPCS.png')

