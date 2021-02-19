# Hamiltonian Monte Carlo simulations of X ~ N(0,S)
# covariance: S = [[1,0.998],[0.998,1]]
# A ~ S^{-1}

import numpy as np
import matplotlib.pyplot as plt

def genTraj(x,_iter):
	path = [x]
	x_values = [x]
	x = np.array(x)
	A = np.array([[250.25, -249.75],[-249.75, 250.25]])
	n = A.shape[0]	

	def evalE(_x):

		print(_x)
		return np.dot(_x,np.dot(A,_x)) / 2.
	
	def evalG(_x):
		return np.dot(A,_x)
	
	epsilon = 0.055
	L = 19
	for i in range(_iter):
		#draw a new p
		p = np.random.randn(n)
		H = np.dot(p,p) / 2. + evalE(x)
	
		x_new = x
		G_new = evalG(x)
        
        # leap frog
		tmp = []
		for j in range(L):
			# Make a half step for momentum at the beginning
			p = p - epsilon*G_new/2.
        	# alternate full steps for position and momentum
			x_new = x_new + epsilon * p
			print(x_new)
			tmp.append([x_new[0],x_new[1]])
			G_new = evalG(x_new)
			p = p - epsilon*G_new/2.
		
		E_new = evalE(x_new)
		H_new = np.dot(p,p) / 2. + E_new
		dH = H - H_new
	
		u = np.random.rand()
		if u < np.exp(dH):
			G = G_new
			x = x_new
			E = E_new
			print(tmp)
			for tau in range(L):
				path.append(tmp[tau])
			x_values.append([x[0],x[1]])

	return np.array(path), np.array(x_values)

def plotTraj(x,v):
	A = np.array([[250.25, -249.75],[-249.75, 250.25]])
	c = 1. / A[0,0]
	b = A[0,1] * c * 2
	xl = np.sqrt( 4 * c / (4 - b**2))
	xp = np.linspace(-xl,xl,200) 
	ypu = (- b * xp + np.sqrt(b**2 * xp**2 + 4 * (c - xp**2))) * 0.5
	ypb = (- b * xp - np.sqrt(b**2 * xp**2 + 4 * (c - xp**2))) * 0.5
	plt.plot(xp,ypu,'b--')
	plt.plot(xp,ypb,'b--')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.xlim(min(-1.,np.min(x[:,0])), max(1.,np.max(x[:,0])))
	plt.ylim(min(-1.,np.min(x[:,1])), max(1.,np.max(x[:,1])))
	plt.plot(x[:,0],x[:,1],'r-')
	plt.plot(v[:,0],v[:,1],'ro')
	plt.show()

# simulate

_iter = input('Enter number of iterations: ')
x,v = genTraj([-0.9,-0.9],int(_iter))
plotTraj(x,v)


