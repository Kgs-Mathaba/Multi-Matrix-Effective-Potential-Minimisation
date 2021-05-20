
import numpy as np
from scipy import linalg
#from itertools import combinations
from scipy.optimize import minimize
from numpy.linalg import multi_dot
from timeit import default_timer as timer
import datetime


import Collective

global loop
global omega
global little_omega
global LnJ
"""# Main Program"""

lmax = 8 
omega_length = int(lmax)
max_size = 2*omega_length-2
omega_size = omega_length
little_omega_size=omega_size-2
c = 0.0
g_3 = 0.0
g_4 = 1.0 
N = 16


print ("Total number of loops is", max_size ,". Omega is a ", omega_size,"x", omega_size,"matrix. " )
print()
print ("Number of master variables :", N)
print()

# intializes loops, omega, little_omega and LnJ with zeros

loop, omega, little_omega, LnJ = Collective.initialize_loops(omega_size, max_size)

print("loop = ", loop)
print()
print("omega = ", omega)
print()
# initial distribution of egenvalues for both matrices
x_init=(np.random.rand(N) - np.ones(N)*0.5)*np.sqrt(N)
print("x_init =", x_init)
print("loop before effect_pot = ", loop)
Hamiltonian = Collective.effect_pot(x_init, omega_length, max_size, little_omega_size, N, c, g_3, g_4)
print("Hamiltonian = ", Hamiltonian)
print("loop after effect_pot = ", loop)

print(x_init.shape)

Collective.effect_pot_grad(x_init, omega_length, max_size, little_omega_size, N, c, g_3, g_4)


start = timer()
Energy = minimize (Collective.effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='BFGS', jac=Collective.effect_pot_grad, options={'disp': True, 'maxiter': 20000,'gtol': np.sqrt(N)*1e-10})#changed from e-16 20/03/21#fixed N 20/04/2021
end = timer()

print ()
print ("g_3 =", g_3, ",","g_4=", g_4 )
print ("No of loops =",max_size, ",","lmax =",omega_length, ",","Omega is", omega_size,"x", omega_size, "matrix")
print ( "No of master variables = ", N)
#np.format_float_scientific(np.amin(np.abs(np.concatenate((grad_1.real,grad_2.real),axis=0))),precision=6)

print("Minimization time",",", datetime.timedelta(seconds = int(end-start)), "(days,h:m:s)")

print("Smallest gradient component modulus:",",",np.format_float_scientific(np.amin(np.abs(Energy.jac)),precision=6))
print("Largest gradient component modulus:",",",np.format_float_scientific(np.amax(np.abs(Energy.jac)),precision=6))

print ()
print("Energy:",",",np.format_float_scientific(Energy.fun,precision=6))
print("Energy:",",",format(Energy.fun.real, '.6f'))

grad_list=np.zeros((max_size+1,N),dtype=complex)
for i in range(1,max_size+1) :
    loop[i]=np.sum (np.power(Energy.x,i)) / (N**((i+2)/2))
    grad_list[i] = i*np.power(Energy.x,i-1) / (N**((i+2)/2))
grad_list [0] = np.zeros(N)
loop[0]=1.0

grad_2_list=np.zeros((max_size+1,N),dtype=complex)
for i in range(2,max_size+1) :
    grad_2_list[i] = i*(i-1)*np.power(Energy.x,i-2) / (N**((i+2)/2))
grad_2_list [0] = np.zeros(N)
grad_2_list [1] = np.zeros(N)

for i in range (max_size+1):
    print ("Loop,", i, ",has value,", format(loop[i].real, '.6f'))#, ",imaginary part,", format(loop[i].imag, '.6f'))#round (loop[i].real,5)

for i in range(omega_size):
    for j in range(omega_size):
        omega[i,j]=(i+1)*(j+1)*loop[i+j]

print("Omega = ", omega)
print()
print ("Is Omega symmetric ? ", np.array_equal (omega, omega.T))
print()


print ("Symmetric Omega eigenvalues")
Eigen = np.sort(linalg.eigvals(omega))
i=0
for x in list(np.sort(linalg.eigvals(omega))):
    print ( np.format_float_scientific(Eigen[i].real ,precision=6))
    i+=1



A=np.zeros((omega_size,omega_size),dtype=complex)
for k in range (omega_size):
    for i in range(1,omega_size):
        for j in range (i):
            A[i,k]+=(i+1)*(Collective.k_delta(j,k+1)*loop[i-1-j] + loop[j]*Collective.k_delta(i-1-j,k+1))


A_2=np.zeros((omega_size,omega_size,omega_size),dtype=complex)
for k in range (omega_size):
    for l in range (omega_size):
        for i in range(1,omega_size):
            for j in range (i):
                A_2[i,k,l]+=(i+1)*(Collective.k_delta(j,k+1)*Collective.k_delta(i-1-j,l+1) + Collective.k_delta(j,l+1)*Collective.k_delta(i-1-j,k+1))


omega_grad=np.zeros((omega_size,omega_size,omega_size),dtype=complex)
for k in range (omega_size):
    for i in range(omega_size):
        for j in range(omega_size):
            omega_grad[i,j,k]+=(i+1)*(j+1)*Collective.k_delta(i+j,k+1)


omega_deriv= np.sum(np.array([[[omega_grad[i,j,k]*LnJ[j] for k in range(omega_size)]for i in range(omega_size)]for j in range(omega_size)]),axis=0)


for i in range(1,omega_size):
    little_omega[i]=0.0
    for j in range(i):
        little_omega [i] += (i+1)*loop[j]*loop[i-1-j]
little_omega[0]=0.0


LnJ = np.linalg.solve(omega, little_omega)



omega_inv=linalg.inv(omega)

V_2=multi_dot([A.T,omega_inv,A])

V_2+=np.sum(np.array([[[A_2[i,k,l]*LnJ[i] for k in range(omega_size)]for l in range(omega_size)]for i in range(omega_size)]),axis=0)



V_2+=multi_dot([omega_deriv.T,omega_inv,omega_deriv])

V_2-=multi_dot([A.T, omega_inv,omega_deriv])+multi_dot([omega_deriv.T, omega_inv,A])

v_2=multi_dot ([omega,V_2])/4.0

print()
print ("Full fluctuations spectrum wrt loops")
for x in list(np.sort(np.sqrt(linalg.eigvals(v_2).real))):
    print (np.format_float_scientific(x,precision=6))

print ()
print ("Full fluctuations spectrum wrt loops - decimal points")
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(v_2).real)))):
    print ("{:.8f}". format(x))


