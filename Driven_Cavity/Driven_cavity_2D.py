'''
problem: Driven cavity (2D) two-dimensional lid-driven cavity problem - 21.4.3 : pag 404 FEniCS book

Specific parameters:
domain: UnitSquare[0,1]^2
kinematic viscosity: 1/1000
No-slip boundary conditions are imposed on each edge of the square, except at the upper edge where the velocity is set to u = ( 1, 0 ).
initial condition for the velocity is set to zero.

Expecting result:
The resulting flow is a vortex developing in the upper right corner and then traveling towards the center of the square as the flow evolves.

Benchmark:

functional of interest: the minimum value of the stream function at final time T = 2.5. Available in Pandit et al. (2007), where a reference value of min Phi = -0.0585236 is reported

'''

from dolfin import *
from chorin import chorin_method
#from CSS import css_method
from stream_function import stream_function
from ipcs import ipcs_method
import numpy as np
import matplotlib.pyplot as plt
import time

teste = 1

# Mesh and function spaces
n_cells = 64	
mesh = UnitSquareMesh(n_cells, n_cells)
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
u_e = -0.0585236

i = teste


inicio1 = time.time()
(u1,p1) = chorin_method(k, u, v, nu, f, p, q, bcu, bcp, dt, V, Q, teste, u0, p0, T)
fim1 = time.time()

print('\n\n\n')

print('Tempo chorin: ', fim1-inicio1)
print('Velocity chorin: ', np.array(u1.vector()).mean())
psi1 = stream_function(u1)
stream_min1 = np.array(psi1.vector()).min()
print('Psi min chorin: %f', stream_min1)
u1_a = stream_min1
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

print('\n\n\n')

print('Tempo IPCS:   ', fim2-inicio2)
print('Velocity IPCS: ', np.array(u2.vector()).mean())
psi2 = stream_function(u2)
stream_min2 = np.array(psi2.vector()).min()
print('Psi min IPCS: ', stream_min2)
u2_a = stream_min2
errorIPCS = 1 - (u_e - u2_a)/u_e
print('degrees of freedom:  ', u2.vector().size())
print('error IPCS: ', errorIPCS)

plot(u2)
plt.savefig('VelocityIPCS'+str(i)+'.png')
plot(p2)
plt.savefig('PressureIPCS'+str(i)+'.png')
print('\n\n\n')

plot(psi1)
plt.savefig('streamChorin'+str(i)+'.png')
plot(psi2)
plt.savefig('streamIPCS'+str(i)+'.png')

