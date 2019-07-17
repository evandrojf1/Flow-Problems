from dolfin import *
import numpy as np
import ufl
import time

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def chorin_method(mesh, c_d, U_e, P_e):
	h = mesh.hmin()
	n = FacetNormal(mesh)

	#Function spaces
	V = VectorFunctionSpace(mesh, 'P', 2, constrained_domain=c_d)
	Q = FunctionSpace(mesh, 'P', 1, constrained_domain=c_d)
	u = TrialFunction(V)
	v = TestFunction(V)
	p = TrialFunction(Q)
	q = TestFunction(Q)

	# Define functions for solutions at previous and current time steps
	u_n = Function(V)
	u_  = Function(V)
	p_n = Function(Q)
	p_  = Function(Q)

	#Fluid properties
	rho = 1
	nu = 1/100

	#Simulation parameters
	Um = 1.
	dt = 0.2*h/Um
	k = dt
	f = Constant((0.,0.))
	T = 0.5

	#initial condiction
	u0 = Expression(U_e, degree=3, t=0, nu=float(nu))
	p0 = Expression(P_e, degree=3, t=0, nu=float(nu))
	u_n = interpolate(u0, V)
	p_n = interpolate(p0, Q)

	# Define symmetric gradient
	def epsilon(u):
		return sym(nabla_grad(u))
	# Define stress tensor
	def sigma(u, p):
		return 2*mu*epsilon(u) - p*Identity(len(u))

	# Tentative velocity step
	F1 = (1/k)*inner(v, u - u_n)*dx + inner(v, grad(u_n)*u_n)*dx \
	+ nu*inner(grad(v), grad(u))*dx - inner(v, f)*dx
	a1 = lhs(F1)
	L1 = rhs(F1)
	# Poisson problem for the pressure
	a2 = inner(grad(q), grad(p))*dx
	L2 = -(1/k)*div(u_)*q*dx
	# Velocity update
	a3 = inner(u, v)*dx
	L3 = inner(u_, v)*dx - k*inner(nabla_grad(p_), v)*dx

	# Assemble matrices
	A1 = assemble(a1)
	A2 = assemble(a2)
	A3 = assemble(a3)

	solver02 = KrylovSolver('bicgstab', 'hypre_amg')

	solver1 = KrylovSolver('cg', 'sor')
	null_vec = Vector(p_n.vector())
	Q.dofmap().set(null_vec, 1.0)
	null_vec *= 1.0/null_vec.norm('l2')
	null_space = VectorSpaceBasis([null_vec])
	as_backend_type(A2).set_nullspace(null_space)

	t = 0
	#for n in range(num_steps):
	inicio = time.time()
	while (t<=T):
		# Compute tentative velocity step
		b1 = assemble(L1)
		solver02.solve(A1, u_.vector(), b1)
		#end()

		# Pressure correction
		b2 = assemble(L2)
		null_space.orthogonalize(b2);
		solver1.solve(A2, p_.vector(), b2)
		#end()

		# Velocity correction
		#begin("Computing velocity correction")
		b3 = assemble(L3)
		solver02.solve(A3, u_.vector(), b3)
		#end()

		# Move to next time step
		# Update previous solution
		u_n.assign(u_)
		p_n.assign(p_)
		#print(t)
		t += dt
	fim = time.time()

	tempo = fim - inicio
	return u_, p_, tempo

