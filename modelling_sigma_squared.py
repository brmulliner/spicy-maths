import numpy as np
from scipy.integrate import quad
from scipy.stats import invgamma
import matplotlib.pyplot as plt  # import to make graphs, use plt


def Sigma(h,phi,mu,n):                    # update method for sigma squared
    a = 0.5*((1-phi**2.)*((h[0]-mu)**2) + np.sum((h[1:]-mu-phi*(h[:-1]-mu))**2))
    #print(a)
    sigma = (invgamma.rvs(n/2, scale=a, size = 1000000))
    return sigma

def sigma_analytical(x,n,a):
    return (x**((-n/2)-1))*np.exp(-a/x)


n = 1000
#h=np.array([-.1]*n)
phi = 0.97
mu = -1.0


sigma = Sigma(h,phi,mu,n)
#print(sigma)






a = 0.5*((1-phi**2.)*((h[0]-mu)**2) + np.sum((h[1:]-mu-phi*(h[:-1]-mu))**2))

#x=np.linspace(0.045,0.055,num=1000)
#N = quad(sigma_analytical,0.04,0.06,args=(n,a))
print("var_a = ",)
print("mean_a = ",)

#plt.plot(x, 1./ N[0]*sigma_analytical(x,n,a))
plt.hist(sigma, bins='auto', density=1)
print("var_hist = ",np.var(sigma))
print("mean_hist = ",np.mean(sigma))

plt.ylabel('P$(\sigma_\eta^2)$')
plt.xlabel('$\sigma_\eta^2$')
#plt.title('$\sigma_\eta^2$ sampling method vs. analytical result')
#plt.savefig('sigma_sample')