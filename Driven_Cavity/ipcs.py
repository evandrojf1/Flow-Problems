from dolfin import *
import numpy as np

def ipcs_method(rho, n, k, u, v, mu, f, p, q, bcu, bcp, dt, V, Q, teste, u0, p0, T):


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
	#p_n = interpolate(p0, Q)
	U = 0.5*(u_n + u)

	#initial condiction
	#u0 = Constant("0.0")
	#u_n = project(u0, V)

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

	
	#save VTK
	ufile = File("Pressure_driven/IPCS/velocity"+str(teste)+".pvd")
	pfile = File("Pressure_driven/IPCS/pressure"+str(teste)+".pvd")
	

	# Create XDMF files for visualization output
	#xdmffile_u = XDMFFile("Flow_cylinder/IPCS/velocity"+str(teste)+".xdmf")
	#xdmffile_p = XDMFFile("Flow_cylinder/IPCS/pressure"+str(teste)+".xdmf")

	t = 0
	#for n in range(num_steps):
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
		
		# Save to file
		#ufile << (u_,t)
		#pfile << (p_,t)
		
		# Save solution to file (XDMF/HDF5)
		#xdmffile_u.write(u_, t)
		#xdmffile_p.write(p_, t)

		# Move to next time step
		# Update previous solution
		u_n.assign(u_)
		p_n.assign(p_)
		print(t)
		t += dt

	# Save to file
	ufile << (u_,t)
	pfile << (p_,t)

	return u_, p_
