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
	inflow = 'near(x[0],0)'
	outflow = 'near(x[0], 2.2)'
	walls = 'near(x[1], 0) || near(x[1], 0.41)'
	cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

	#inflow_profile = Expression(('4.0*1.5*x[1]*(0.41 - x[1])*sin(3.1415*t/8) / pow(0.41, 4)', '0'), t=0.0, degree=4)
	inflow_profile2 = Expression(('4.0*x[1]*(0.41 - x[1])*1.5*sin(3.14*t/8) / pow(0.41, 2)', '0'), t=0.0, degree=2)

	bcu_inflow = DirichletBC(W.sub(0), inflow_profile2, inflow)
	bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
	bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
	bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)
	bcs = [bcu_inflow, bcu_walls, bcu_cylinder, bcp_outflow]

	#Trial and test functions
	vp = TrialFunction(W)
	u, p = ufl.split(vp)
	#u, p = TrialFunctions(W)
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
	dt = 0.2*h/10.
	idt = 1./dt
	t_end = 0.8
	nu = 1/1000
	f = Constant((0.,0.))
	n = FacetNormal(mesh)
	rho = 1.
	print('dt: ', dt)
	print('\n\nNum iteracoes: ', int(t_end/dt))

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

	linear = {tolr: 1.0E-8, tola: 1.0E-6, mi: 10000, enon: False}
	nonlinear = {tolr: 1.0E-4, tola: 1.0E-4, ln:'gmres', mi:20, ln:'gmres', prec:'ilu', enon: False, ks: linear}
	par = {ns: nonlinear}

	# Time-stepping
	t = 0

	inicio = time.time()
	while t < t_end:
		inflow_profile2.t = t*10
		print ("t = ", t)
		# Compute
		#begin("Solving ....")
		solve(F1 == 0, vp1, bcs=bcs, J=J, solver_parameters = par)
		#end()
		# Extract solutions:
		(u1, p1) = vp1.split(True)
		
		# Move to next time step
		un.assign(u1)
		t += dt
	fim = time.time()

	tempo = fim-inicio

	return u1, p1, tempo
