from dolfin import *
import numpy as np
import ufl
import time

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def ipcs_method(mesh):
	#Fluid properties
	rho = 1
	nu = 1/1000

	h = mesh.hmin()
	V = VectorFunctionSpace(mesh, 'P', 2)
	Q = FunctionSpace(mesh, 'P', 1)
	n = FacetNormal(mesh)

	# Define Boundaries and Boundary Condictions
	inflow = 'near(x[0],0)'
	outflow = 'near(x[0], 2.2)'
	walls = 'near(x[1], 0) || near(x[1], 0.41)'
	cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

	#inflow_profile1 = Expression(('2.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0'), degree=2)
	inflow_profile2 = Expression(('4.0*x[1]*(0.41 - x[1])*1.5*sin(3.14*t/8) / pow(0.41, 2)', '0'), t=0.0, degree=2)
	#inflow_profile2 = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

	#bcu_inflow = DirichletBC(V, Expression(inflow_profile2, degree=2), inflow)
	bcu_inflow = DirichletBC(V, inflow_profile2, inflow)
	bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
	bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
	bcp_outflow = DirichletBC(Q, Constant(0), outflow)
	bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
	bcp = [bcp_outflow]

	# Define variacional parameters
	u = TrialFunction(V)
	p = TrialFunction(Q)
	v = TestFunction(V)
	q = TestFunction(Q)

	# Set parameter values
	T = 0.8           # final time
	dt = 0.2*h/8. # time step CFL with 1 = max. velocity

	print('\n\n ', dt)

	k = dt
	u0 = Constant((0.0, 0.0))
	p0 = Constant(0.0)
	f = Constant((0.0, 0.0))

	# Define functions for solutions at previous and current time steps
	u_n = Function(V)
	u_  = Function(V)
	p_n = Function(Q)
	p_  = Function(Q)

	#initial condiction
	#u_n = project(u0, V)
	#p_n = project(p0, Q)
	U = 0.5*(u_n + u)

	# Define symmetric gradient
	def epsilon(u):
		return sym(nabla_grad(u))
	# Define stress tensor
	def sigma(u, p):
		return 2*nu*epsilon(u) - p*Identity(len(u))

	# Tentative velocity step
	F1 = rho*dot((u - u_n) / k, v)*dx \
	+ rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
	+ inner(sigma(U, p_n), epsilon(v))*dx \
	+ dot(p_n*n, v)*ds - dot(nu*nabla_grad(U)*n, v)*ds \
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

	t = 0
	inicio = time.time()
	while(t<= T):
		inflow_profile2.t = t*10
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

		u_n.assign(u_)
		p_n.assign(p_)
		print(t)
		t += dt

	fim = time.time()

	tempo = fim-inicio

	return u_, p_, tempo

