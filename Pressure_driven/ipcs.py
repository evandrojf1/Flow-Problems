from dolfin import *
import numpy as np
import ufl
import time

def ipcs_method(mesh):
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

	mu = 1/8             # kinematic viscosity
	rho = 1
	k = dt
	u0 = Constant((0.0, 0.0))
	p0 = Constant(0.0)
	f = Constant((0.0, 0.0))

	# Define symmetric gradient
	def epsilon(u):
		return sym(nabla_grad(u))
	# Define stress tensor
	def sigma(u, p):
		return 2*mu*epsilon(u) - p*Identity(len(u))

	# Define functions for solutions at previous and current time steps
	u_n = Function(V)
	u_  = Function(V)
	p_n = Function(Q)
	p_  = Function(Q)

	u_n = interpolate(u0, V)
	U = 0.5*(u_n + u)

	# Tentative velocity step
	F1 = rho*dot((u - u_n) / k, v)*dx \
	+ rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
	+ inner(sigma(U, p_n), epsilon(v))*dx \
	+ dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
	- dot(f, v)*dx
	a1 = lhs(F1)
	L1 = rhs(F1)
	# Poisson problem for the pressure
	a2 = dot(nabla_grad(p), nabla_grad(q))*dx
	L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
	# Velocity update
	a3 = dot(u, v)*dx
	L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

	# Assemble matrices
	A1 = assemble(a1)
	A2 = assemble(a2)
	A3 = assemble(a3)

	# Apply boundary conditions to matrices
	[bc.apply(A1) for bc in bcu]
	[bc.apply(A2) for bc in bcp]

	i = 0
	t = 0
	inicio = time.time()
	while (t<=T):
		#inflow_profile2.t = t
		# Compute tentative velocity step
		#begin("Computing tentative velocity")
		b1 = assemble(L1)
		[bc.apply(b1) for bc in bcu]
		solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')
		#end()

		# Pressure correction
		#begin("Computing pressure correction")
		b2 = assemble(L2)
		[bc.apply(b2) for bc in bcp]
		solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
		#end()

		# Velocity correction
		#begin("Computing velocity correction")
		b3 = assemble(L3)
		solve(A3, u_.vector(), b3, 'cg', 'sor')
		#end()

		# Move to next time step
		# Update previous solution
		u_n.assign(u_)
		p_n.assign(p_)
		#print(t)
		t += dt
	fim = time.time()
	tempo = fim-inicio

	return u_,p_,tempo
