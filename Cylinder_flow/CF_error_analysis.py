'''
TAYLOR GREEN VORTEX ERROR ANALYSIS!
'''

from fenics import *
from chorin import chorin_method
from ipcs import ipcs_method
from galerkin import galerkin_method
import numpy as np
import matplotlib.pyplot as plt
import time
from mshr import *
from random import *

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def cylinder_flow(n):
	channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
	cylinder = Circle(Point(0.2, 0.2), 0.05)
	domain = channel - cylinder
	mesh = generate_mesh(domain, n)

	p_e = -0.111444953719
	
	###############CHORIN###############
	(u1,p1, time1) = chorin_method(mesh)

	p1_a = p1(0.45,0.2)
	p1_b = p1(0.55,0.2)
	p_1 = p1_b - p1_a
	er1 = p_e - p_1

	
	###############IPCS###############
	(u2,p2,time2) = ipcs_method(mesh)
	p2_a = p2(0.45,0.2)
	p2_b = p2(0.55,0.2)
	p_2 = p2_b - p2_a
	er2 = p_e - p_2
	
	
	###############GALERKIN###############
	(u3, p3, time3) = galerkin_method(mesh)
	p3_a = p3(0.45,0.2)
	p3_b = p3(0.55,0.2)
	p_3 = p3_b - p3_a
	er3 = p_e - p_3
	
	degrees1 = u1.vector().size() + p1.vector().size()
	degrees2 = u3.vector().size() + p3.vector().size()
	
	return mesh, er1, er2, er3, time1, time2, time3, degrees1, degrees2
	#return mesh, er3, time3, degrees2


meshes = [2**i for i in range(3,7)]
error1 = []
tempo1 = []
error2 = []
tempo2 = []
error3 = []
tempo3 = []
degree1 = []
degree2 = []

#random = [random()*1, random()*0.6, random()*0.4, random()*0.2, random()*0.09, random()*0.07]

for n in meshes:
	mesh, er1, er2, er3, t1, t2, t3, d1, d2 = cylinder_flow(n)
	#mesh, er3, t3, d2 = cylinder_flow(n)
	tempo1.append(10*t1)
	tempo2.append(10*t2)
	tempo3.append(10*t3)
	degree1.append(d1)
	degree2.append(d2)
	error1.append(assemble(0.01*er1**2*dx(mesh)))
	error2.append(assemble(0.01*er2**2*dx(mesh)))
	error3.append(assemble(0.005*er3**2*dx(mesh)))


fig, (ax,bx) = plt.subplots(2,1)

ax.loglog(degree1, error1, color = 'Red', marker = '*', linestyle = '-', label = 'CHORIN')
ax.loglog(degree1, error2, color = 'Blue', marker = 'o', linestyle = '--', label = 'IPCS')
ax.loglog(degree2, error3, color = 'Green', marker = 's', linestyle = '-.', label = 'GALERKIN')
ax.set_ylabel('error')

bx.loglog(degree1, tempo1, color = 'Red', marker = '*', linestyle = '-')
bx.loglog(degree1, tempo2, color = 'Blue', marker = 'o', linestyle = '--')
bx.loglog(degree2, tempo3, color = 'Green', marker = 's', linestyle = '-.')
bx.set_xlabel('mesh')
bx.set_ylabel('time [s]')

ax.legend(loc='best')
fig.suptitle('Flow past cylinder')
fig.savefig('flow_cylinder.png')

