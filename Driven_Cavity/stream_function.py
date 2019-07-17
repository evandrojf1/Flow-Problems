from dolfin import *
import numpy as np

def stream_function(u):

	class BoundaryValue(Expression):
		def eval(self, values, x):
			if x[0] > DOLFIN_EPS and x[0] < 1.0 - DOLFIN_EPS and x[1] > 1.0 - DOLFIN_EPS:
				values[0] = 1.0
				values[1] = 0.0
			else:
				values[0] = 0.0
				values[1] = 0.0


	# Define variational problem
	V = u.function_space().sub(0).collapse()
	psi = TrialFunction(V)
	q = TestFunction(V)
	a = inner(grad(psi), grad(q))*dx
	L = inner(u[1].dx(0) - u[0].dx(1), q)*dx
	# Define boundary condition
	g = Constant(0)
	bc = DirichletBC(V, g, DomainBoundary())
	# Compute solution
	psi = Function(V)
	solve(a == L, psi, bc)

	return psi
