from dolfin import *
import numpy as np

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def chorin_method(k, u, v, nu, f, p, q, bcu, bcp, dt, V, Q, i, u0, p0, T):


	# Define functions for solutions at previous and current time steps
	u_n = Function(V)
	u_  = Function(V)
	p_n = Function(Q)
	p_  = Function(Q)

	#initial condiction
	u_n = project(u0, V)
	#p_n = project(p0, Q)

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

	# Apply boundary conditions to matrices
	[bc.apply(A1) for bc in bcu]
	[bc.apply(A2) for bc in bcp]

	#save VTK
	#ufile = File("Pressure_driven/Chorin/velocity"+str(i)+".pvd")
	#pfile = File("Pressure_driven/Chorin/pressure"+str(i)+".pvd")

	t = 0
	while(t<= T):

		u_.t = t
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
	
		# Save to file
		#ufile << u_
		#pfile << p_

		# Move to next time step
		# Update previous solution
		u_n.assign(u_)
		p_n.assign(p_)
		#print(t)
		t += dt

	return u_, p_

