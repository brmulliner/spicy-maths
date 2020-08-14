import matplotlib.pyplot as plt  # import to make graphs, use plt
import numpy as np # type np to use numpy
#import math # able to use maths equations
#import statistics as s # able to use variance

def SV_time_series(sigma,mu,phi,n):
    rv = np.random.normal(0,np.sqrt(sigma),n)     
    v = np.zeros(n)                             # v is volatility
    y = np.zeros(n)                             # y is the time series
    v[0] = mu + rv[0]                           # initial v
    for i in range(1, n):
        v[i] = mu + phi *(v[i-1] - mu) + rv[i]
        ry = np.random.normal(0,1.0,n)
    y = np.exp(v/2) * ry                # stochastic volatility time series, y_t
    return y, v

def ACF(t,x,N):
    mean_x = np.mean(x)
    var_x = np.var(x)
    a = (x[:(N-t)]-mean_x)*(x[t:]-mean_x)
    b = (1/N) * np.sum(a) / var_x
    return b


n = 1000                          # n is length of the time series
sigma = 0.05
mu = -1
phi = 0.97

y, v = SV_time_series(sigma,mu,phi,n)

acf_t = np.arange(0,21,1)
acf = np.zeros(len(acf_t))
for i in range(len(acf)):
    acf[i] = ACF(acf_t[i],v,n)

plt.plot(acf_t,acf)
plt.ylabel("$ACF(t)$")
plt.xlabel("$t$")
plt.show()


#plt.plot(range(n),y)
#plt.plot(range(n),v)
#plt.ylabel('$y_t$')
#plt.xlabel('t')
#plt.savefig('yt_test_func.png')

#plt.plot(range(n),v)


"""
t = 100000
mu = np.linspace(0,1,n)
mu_plot = np.array([])
for i in range(n):
    rv = np.random.normal(0,np.sqrt(0.1),t)     
    v = rv                              # v is volatility
    v[0] = 0
    v[1:] = mu[i] + 0.1 *(v[0:-1] +1) + rv[1:]
    ry = np.random.normal(0,0.1,t)
    y = np.array(ry)
    y = np.exp(v/2) * ry
    vy = s.variance(y)
    mu_plot = np.append(mu_plot,vy)
    
phi = np.linspace(0,1,n)
phi_plot = np.array([])
for i in range(n):
    rv = np.random.normal(0,np.sqrt(0.1),t)     
    v = rv                              # v is volatility
    v[0] = 0
    v[1:] = 0.1 + phi[i] *(v[0:-1] +1) + rv[1:]
    ry = np.random.normal(0,0.1,t)
    y = np.array(ry)
    y = np.exp(v/2) * ry
    vy = s.variance(y)
    phi_plot = np.append(phi_plot,vy)
    
    
sig = np.linspace(0,1,n)
sig_plot = np.array([])
for i in range(n):
    rv = np.random.normal(0,np.sqrt(sig[i]),t)     
    v = rv                              # v is volatility
    v[0] = 0
    v[1:] = 0.1 + 0.1 *(v[0:-1] +1) + rv[1:]
    ry = np.random.normal(0,0.1,t)
    y = np.array(ry)
    y = np.exp(v/2) * ry
    vy = s.variance(y)
    sig_plot = np.append(sig_plot,vy)
    




#plt.plot(t,v)


plt.plot(mu,mu_plot, label='$\mu$')
plt.plot(mu,phi_plot, label='$\phi$')
plt.plot(mu,sig_plot,label='$\sigma_\eta^2$')
plt.legend()
plt.ylabel('variance of $y_t$')
plt.xlabel('value of parameter being modified')
#plt.title("Graph to show time series' variance as a function of $\mu,\phi, \sigma_\eta^2$")
#plt.savefig('yt_variance.png')
plt.show()

#vh = s.variance(h)
vy = s.variance(y)
my = s.mean(y)
#mh = s.mean(h)

#print("variance h = ",vh)
print("variance y = ",vy)
print("meany = ",my)
#print("meanh = ",mh)
"""