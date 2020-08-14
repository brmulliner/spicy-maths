import matplotlib.pyplot as plt  # import to make graphs, use plt
import numpy as np # type np to use numpy
from scipy.integrate import quad
n1 = 100 # sets N
n2 = range(n1) # sets the region 0 to N
n = 100
h = np.random.normal(-0.1,0,n)
sigma = 1 # parameter sigma
mu = 0


def Phi(h,sigma,phi_arg,mu):                    # update method for phi
    phi_f = phi_arg
    d = -(h[0]-mu)**2 + np.sum((h[0:-1]-mu)**2)
    e = np.sum((h[1:]-mu)*(h[0:-1]-mu))
    while 1==1:
        phi_new =  np.random.normal(e/d,np.sqrt(sigma/d))
        if abs(phi_new) < 1:                     #restricting absolute value
         break
    Pmh = np.min([1.,np.sqrt(1-phi_new**2)/np.sqrt(1-phi_f**2)])
    if np.random.uniform()<=Pmh:
            phi_f = phi_new
    return phi_f


histogram = np.array([])
phi = 0.95
for i in range(100000):
    phi = Phi(h,sigma,phi,mu)
    histogram = np.append(histogram,phi)



d = -(h[0]-mu)**2 + np.sum((h[0:-1]-mu)**2)
e = np.sum((h[1:]-mu)*(h[0:-1]-mu))

x = np.linspace(0.5,1,1000) # creates range of 100 x values
y = ((1-x**2)**0.5) * np.exp((-d/(2*sigma))*(x-(e/d))**2) # equation for P(phi)
def phi_analytical(x,d,e,sigma):
    return ((1-x**2)**0.5) * np.exp((-d/(2*sigma))*(x-(e/d))**2)
N = quad(phi_analytical,-1,1,args=(d,e,sigma))


normy = [float(i)/sum(y) for i in y] # normalises all y values

v1 = np.var(histogram)
v2 = np.var(y)
mean = np.mean(histogram)
print(v1)
print(v2)
print("mean = ",mean)


plt.plot(x,1./N[0]*phi_analytical(x,d,e,sigma))
plt.hist(histogram, bins='auto', density='true')
plt.ylabel('P$(\phi)$')
plt.xlabel('$\phi$')
#plt.title('$\phi$ sampling method vs. analytical result')
#plt.savefig('Phi_modelling')
# plt.show()

