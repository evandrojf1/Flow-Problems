from dolfin import *
import numpy as np
import ufl
import time

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def galerkin_method(mesh):
	# Mesh and function spaces
	h = mesh.hmin()

	V = VectorElement("Lagrange", triangle, 1)
	P = FiniteElement("Lagrange", triangle, 1)
	VP = MixedElement([V,P])
	W = FunctionSpace(mesh, VP)

	# Define Boundaries and Boundary Condictions
	inflow = 'near(x[0], 0)'
	outflow = 'near(x[0], 1)'
	walls   = 'near(x[1], 0) || near(x[1], 1)'

	bcu_noslip  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
	bcp_in  = DirichletBC(W.sub(1), Constant(1.0), inflow)
	bcp_out = DirichletBC(W.sub(1), Constant(0.0), outflow)
	bcs = [bcu_noslip, bcp_in, bcp_out]

	#Trial and test functions
	vp = TrialFunction(W)
	u, p = ufl.split(vp)
	w, q = TestFunctions(W)

	#Functions
	v0 = Constant((0.,0.))
	vp1 = Function(W)
	u1, p1 = vp1.split()
	un = interpolate(v0, W.sub(0).collapse())

	#Stress tensor
	def epsilon(v):
		return sym(nabla_grad(v))

	def sigma(v,p,nu):
		return 2*nu*epsilon(v)-p*Identity(len(v))

	#Parameters
	dt = 0.2*h/1.
	idt = 1./dt
	t_end = 0.5
	nu = 1/8
	f = Constant((0.,0.))
	n = FacetNormal(mesh)
	rho = 1.

	#Standard Galerkin
	F = (1.0/dt)*inner(u-un,w)*dx + inner(dot(u1,nabla_grad(u)),w)*dx + inner(sigma(u,p,nu),epsilon(w))*dx + q*div(u)*dx - inner(f,w)*dx + inner(p*n,w)*ds - nu*inner(w,grad(u).T*n)*ds

	##########STABILIZATION#############
	h = 2.0*Circumradius(mesh)
	unorm = sqrt(dot(u1, u1))

	#Residuo momento
	R = (1.0/dt)*(u-un)+dot(u1,nabla_grad(u))-div(sigma(u,p,nu)) + nabla_grad(p)

	#PSPG
	tau_pspg = (h*h)/2.
	F_pspg = tau_pspg*inner(R,nabla_grad(q))*dx

	#SUPG
	tau_supg = ((2.0*idt)**2 + (2.0*u/h)**2 + 9.0*(4.0*nu/h**2)**2 )**(-0.5)
	F_supg = tau_supg*inner(R,dot(u1,nabla_grad(w)))*dx

	#LSIC
	tau_lsic = 0.5*unorm*h
	F_lsic = inner(div(u),div(w))*tau_lsic*dx

	F_est = F + F_pspg + F_supg + F_lsic
	#F_est = F + F_pspg + F_supg

	# define Jacobian
	F1 = action(F_est,vp1)
	J = derivative(F1, vp1,vp)

	#solution parameters
	tolr = 'relative_tolerance'
	tola = 'absolute_tolerance'
	ln = 'linear_solver'
	ns = 'newton_solver'
	prec = 'preconditioner'
	ks = 'krylov_solver'
	mi = 'maximum_iterations'
	enon = 'error_on_nonconvergence'

	linear = {tolr: 1.0E-6, tola: 1.0E-6, mi: 1000, enon: False}
	nonlinear = {tolr: 1.0E-5, tola: 1.0E-5, ln:'gmres', mi:3, ln:'gmres', prec:'ilu', enon: False, ks: linear}
	par = {ns: nonlinear}


	# Time-stepping
	t = 0
	# Create files for storing solution
	#ufile = File("Driven_cavity/velocity"+str(n_cells)+".pvd")
	#pfile = File("Driven_cavity/pressure"+str(n_cells)+".pvd")

	inicio = time.time()
	while t < t_end:
		print ("t = ", t)
		# Compute
		#begin("Solving ....")
		solve(F1 == 0, vp1, bcs=bcs, J=J, solver_parameters = par)
		#end()
		# Extract solutions:
		(u1, p1) = vp1.split(True)
		# Plot
		#plot(v)
		# Save to file
		#ufile << u1
		#pfile << p1
		# Move to next time step
		un.assign(u1)
		t += dt
		#print('velocity no final: ', np.array(v.vector[0]).mean())
	fim = time.time()

	tempo = fim-inicio

	return u1, p1, tempo
