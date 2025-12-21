import numpy as np


s = np.linspace(0,100,101)
shape = (2,101) +(1,)*3

x = np.array([np.cos(s),np.sin(s)])

N_G = 9
print(x.shape)
l= x.reshape(shape)
print(s.shape)
print(shape)
print(l.shape,l)
coordoriginal, woriginal = np.polynomial.legendre.leggauss(N_G)
coordquad = (coordoriginal + 1) / 2
weights = woriginal / 2
print(coordquad,weights)
coordquad = coordquad.reshape((1, 1, N_G))
weights = weights.reshape((1, 1, N_G))

print(coordquad,weights)

dist = np.linalg.norm(rho_m.reshape(rho_m.shape + (1, 1, 1)) - rho_p,axis=0
        ) 
