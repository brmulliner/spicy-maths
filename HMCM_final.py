import numpy as np
from scipy.stats import invgamma
import matplotlib.pyplot as plt  # import to make graphs, use plt


"""
Generates the SV time series y_t
"""
def SV_time_series(y_sigma,y_mu,y_phi,n):
    rv = np.random.normal(0,np.sqrt(y_sigma),n)     
    v = np.zeros(n)                             # v is volatility
    y = np.zeros(n)                             # y is the time series
    v[0] = y_mu + rv[0]                           # initial v
    for i in range(1, n):
        v[i] = y_mu + y_phi *(v[i-1] - y_mu) + rv[i]
        ry = np.random.normal(0,1.0,n)
    y = np.exp(v/2) * ry                # stochastic volatility time series, y_t
    return y


"""
Sigma^2 update scheme
"""
def Sigma(h,phi,mu,n):                    # update method for sigma squared
    a = 0.5*((1-phi**2.)*((h[0]-mu)**2) + np.sum((h[1:]-mu-phi*(h[:-1]-mu))**2))
    sigma = (invgamma.rvs(n/2, scale=a, size = 1))
    return sigma[0]


"""
Mu update scheme
"""
def Mu(h,sigma, phi,n):                   # update method for mu
    b = (1-phi**2) + (n-1)*((1-phi)**2)
    c = (1-phi**2)*h[0] + (1-phi)*(np.sum(h[1:] - phi*h[:-1]))
    mu = np.random.normal(c/b,np.sqrt(sigma/b))
    return mu

"""
Phi update scheme
"""
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


"""
Calculates the value of the Hamiltonian
"""
def H(p,h,y,phi,sigma,mu): # defines the function for the Hamiltonian
    A =  0.5*np.sum(p**2) 
    B =  np.sum(h/2 + ((y**2)/2)*np.exp(-h))
    C =  (((h[0]-mu)**2)*(1-(phi**2)))/(2*(sigma))
    D =  np.sum(((h[1:]-mu-phi*(h[:-1]-mu))**2)/(2*(sigma)))
    return A+B+C+D


"""
Differentiaties the Hamiltonian with respect to h
"""
def dH(h,y,phi,sigma,mu): # defines the derivative dH/dh for i= 1 to N
    dh = np.zeros(len(h)) # creates empty array of length h_t
 
    #exponential terms for i = 1, 2-(n-1), n
    B0 = 0.5-((y[0]   **2)/2)*np.exp(-h[0]   )
    B_ = 0.5-((y[1:-1]**2)/2)*np.exp(-h[1:-1])
    B1 = 0.5-((y[-1]  **2)/2)*np.exp(-h[-1]  )
    
    #(h_1 - mu) terms for i = 1, 2-(n-1), n
    C0 = ((h[0]   -mu)*(1.-phi**2.))/(sigma)
    C_ = 0
    C1 = 0
    
    #final terms for i = 1,2-(n-1),n
    D0 =                                  - phi*(h[1] -mu-phi*(h[0]   -mu)) /(sigma)
    D_ = ((h[1:-1]-mu -phi*(h[0:-2]-mu))  - phi*(h[2:]-mu-phi*(h[1:-1]-mu)))/(sigma)
    D1 = ( h[-1]  -mu- phi*(h[-2]-mu))                                      /(sigma)
    
    #sum terms to generate dH/dh for all i
    dh[0]    = B0 + C0 + D0     #special case i=1
    dh[1:-1] = B_ + C_ + D_     # 2 <= i < n
    dh[-1]   = B1 + C1 + D1     #special case i=n
    
    return dh


"""
Evolves a half step of h - within the leapfrog integrator
"""  
def evolve_hhalf(p,h,dt): # making a function for xnew within the leapfrog method
    hnew = h + (dt/2.0)*p
    return hnew

"""
Evolves a full step of h - within the leapfrog integrator
"""
def evolve_h(p,h,dt): # making a function for xnew within the leapfrog method
    hnew = h + dt*p
    return hnew

"""
Evolves p - within the leapfrog integrator
"""
def evolve_p(p,dt,h,y,phi,sigma,mu): # making a function for pnew within the leapfrog method
    dh = dH(h,y,phi,sigma,mu)
    #print("dH = ",dh[5])
    pnew = p - (dt * dh) # call function for dH
    #print("p_intermediate = ",p)
    return pnew


"""
Performs the 2nd order leapfrog integrator for a trajectory of 1, over 1/dt steps
"""
def HMC_trajectory(p_in,h_in,dt,y,phi,sigma,mu):
    p = p_in
    h = h_in
    h = evolve_hhalf(p,h,dt)              # half step for h
    for j in range(int(1./dt)-1):                 # number of leapfrog repetitions int(1./dt)
        p = evolve_p(p,dt,h,y,phi,sigma,mu)    # p step
        h = evolve_h(p,h,dt)            # half h step
    p = evolve_p(p,dt,h,y,phi,sigma,mu)
    h = evolve_hhalf(p,h,dt)
    return p,h


"""
AUTOCORRELATION FUNCTION
"""
def ACF(t,x,N):
    mean_x = np.mean(x)
    var_x = np.var(x)
    a = (x[:(N-t)]-mean_x)*(x[t:]-mean_x)
    b = (1/N) * np.sum(a) / var_x
    return b





""" GENERATING THE SV TIME SERIES"""
n = 1000            #length of time series
y_sigma = 0.05     
y_mu = -1.0
y_phi = 0.97
#values for volatility parameters within SV
y = SV_time_series(y_sigma,y_mu,y_phi,n) # generates time series

"""SETTING INITIAL CONDITIONS FOR HMCM"""
h_in = np.ones(n)
sigma_in = 1.
mu_in = -0.0
phi_in = 0.5
dt = 0.05
t = 100000

"""SETTING UP PLOTS"""
h_plot = np.zeros(t)
sigma_plot = np.zeros(t+1)
mu_plot = np.zeros(t+1)
phi_plot = np.zeros(t+1)
sigma_plot[0] = sigma_in
mu_plot[0] = mu_in
phi_plot[0] = phi_in


"""HMCM"""
count = 0
for i in range(t):
    p_in = np.random.normal(0,1,n)
    sigma_2 = Sigma(h_in,phi_in,mu_in,n)
    mu_2 = Mu(h_in,sigma_in,phi_in,n)
    phi_2 = Phi(h_in,sigma_in,phi_in,mu_in) 
    H_in = H(p_in,h_in,y,phi_2,sigma_2,mu_2)
    #print("in = ", H_in)
    #print(H(p,h,y,phi,sigma,mu))
    #print(sigma,mu,phi)
    pnew,hnew = HMC_trajectory(p_in,h_in,dt,y,phi_2,sigma_2,mu_2)
    r1 = np.random.uniform(0,1)     # generate flat random number
    H_final = H(pnew,hnew,y,phi_2,sigma_2,mu_2) # final Hamiltonian value
    DH = H_final - H_in           # delta H value
    if r1 <= np.exp(-DH):            # acceptance
        h_in = hnew
        count = count + 1
    sigma_in = sigma_2
    mu_in = mu_2
    phi_in = phi_2  
    sigma_plot[i+1] = sigma_in
    mu_plot[i+1] = mu_in
    phi_plot[i+1] = phi_in
    h_plot[i] = h_in[100]
    

#print("sigma", np.mean(sigma_plot[2000:]))
#print("mu = ", np.mean(mu_plot[2000:]))
#print("phi = ", np.mean(phi_plot[2000:]))

#print("sigma var = ", np.sqrt(np.var(sigma_plot[20000:])))
#print("mu  var = ", np.sqrt(np.var(mu_plot[20000:])))
#print("phi var = ", np.sqrt(np.var(phi_plot[20000:])))


 
acf_t = np.arange(0,21,1)
acf = np.zeros(len(acf_t))
for i in range(len(acf)):
    acf[i] = ACF(acf_t[i],h_plot,t)


   
    
"""PRINTS AND PLOTS"""
plt.plot(acf_t,acf)
plt.ylabel("$ACF(t)$")
plt.xlabel("$t$")
plt.xscale("log")
plt.show()

#print("count = ", count)

#plt.plot(range(t+1),sigma_plot, label = "$\sigma^2$")
#plt.plot(range(t+1),mu_plot, label = "$\mu$")
#plt.plot(range(t+1),phi_plot, label = "$\phi$")
#plt.xlabel("$N_{MC}$")
#plt.ylabel("$\phi$")
#plt.legend()
#print("sigma squ#plt.plot(range(1200),sigma_plot, color = "b")ared = ", np.mean(sigma_plot[10000:]))
#print("mu = ", np.mean(mu_plot[10000:]))
#print("phi = ", np.mean(phi_plot[10000:]))
"""        
plt.plot()
plt.yscale("")
plt.xscale("")
plt.ylabel("")
plt.xlabel()
#plt.title("Continuum Limit")
plt.show()

#print(HMC_trajectory(p,h,dt,y,phi,sigma,mu))
#print(H(p,h,y,phi,sigma,mu))
#print("pnew=",pnew)
#print("hnew=",hnew)
"""



