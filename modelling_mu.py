import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def Mu(h,sigma, phi,n):                   # update method for mu
    b = (1-phi**2) + (n-1)*((1-phi)**2)
    c = (1-phi**2)*h[0] + (1-phi)*(np.sum(h[1:] - phi*h[:-1]))
    #print(b)
    #print(c)
    mu = np.random.normal(c/(b),np.sqrt(sigma/b), size=100000)
    return mu

def mu_analytical(x,b,c,sigma):
    return np.exp((-b/(2.*sigma))*(x-c/b)**2.)

h0 = 0 # starting value for h

n = 1000                          # n is length of the time series
sigma = 0.05
mu = -1
phi = 0.97

rv = np.random.normal(0,np.sqrt(sigma),n)     
h = np.zeros(n)                             # v is volatility
h[0] = mu + rv[0]
for i in range(1, n):
    h[i] = mu + phi *(h[i-1] - mu) + rv[i]




n = 1000
#h=np.random.normal(-0.1,0,n)
phi = 0.97
sigma = 0.05

mu=Mu(h,sigma, phi,n)


#plt.plot(range(n),mu)
b = (1-phi**2) + (n-1)*((1-phi)**2)
c = (1-phi**2)*h[0] + (1-phi)*(np.sum(h[1:] - phi*h[:-1]))

x = np.linspace(-2,-0,1000)


N = quad(mu_analytical,-3.,3.,args=(b,c,sigma))


#print("var_a = ",np.var(x))
#print("mean_a = ",np.mean(x))
plt.plot(x, 1./ N[0]*mu_analytical(x,b,c,sigma))
plt.hist(mu, bins='auto', density=1)
print("var_hist = ",np.var(mu))
print("mean_hist = ",np.mean(mu))

plt.ylabel('P$(\mu)$')
plt.xlabel('$\mu$')
#plt.title('$\mu$ sampling method vs. analytical result')
#plt.savefig('mu_sample')