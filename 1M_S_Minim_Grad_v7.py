



import numpy as np
from scipy import linalg
#from itertools import combinations
from scipy.optimize import minimize
from numpy.linalg import multi_dot
from timeit import default_timer as timer
import datetime

def k_delta (i_1,i_2):

    k_delta_result=0
    if (i_1==i_2) :  k_delta_result=1

    return k_delta_result

def effect_pot(x_N_2, omega_dim, num_loops,litte_omega_dim, nt_hooft, c_coupling, g3_coupling,g4_coupling):

    global loop
    global omega
    global little_omega
    global LnJ

    print ()
    print ("Function call")
    #func_start= timer()

    # Obtain Loops from eigenvalues.
    for i in range(1,num_loops+1) :
        if i< omega_dim:
            loop[i]=np.sum (np.power(x_N_2,i)) / (nt_hooft**((i+2)/2))
        else:
            loop[i]=0
    loop[0]=1.0

    for i in range (max_size+1):
        print ("Loop[",i, "] = ", format(loop[i].real, '.6f'))

    # Generate Omega matrix from Loops
    #omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range(omega_dim):
        for j in range(omega_dim):
            omega[i,j]=(i+1)*(j+1)*loop[i+j]

    print("omega in effect_pot = ")
    print(omega)
    # Generate "little omega" (omeg) from Loops
    #little_omega=np.zeros ((omega_dim),dtype=complex)
    for i in range(1,omega_dim):
        little_omega[i]=0.0
        for j in range(i):
            little_omega [i] += (i+1)*loop[j]*loop[i-1-j]
    little_omega[0]=0.0
    
    # Solve system of linear equations for LnJ, instead of inverting Omega
    LnJ = np.linalg.solve(omega, little_omega)

    # Obtain value of collective effective potential
    large_n_energy = np.dot(little_omega , LnJ)/8.0 + loop[2]/2.0+g3_coupling*loop[3]/3.0+g4_coupling*loop[4]

    print ("Large N Energy :")
    #print (np.format_float_scientific(large_n_energy,precision=6))#doesn't print imaginary part
    print (large_n_energy)
    #func_end= timer()
    #print("Time taken",",", datetime.timedelta(seconds = int(func_end-func_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))

    return large_n_energy.real


def effect_pot_grad(x_N_2, omega_dim, num_loops,litte_omega_dim, nt_hooft, c_coupling, g3_coupling, g4_coupling):

    global loop
    global omega
    global little_omega
    global LnJ
    global grad_list

    print()
    print ("Gradient function call")
    #grad_start= timer()


    # Obtain Loops and derivatives from eigenvalues.
    grad_list=np.zeros((num_loops+1,nt_hooft),dtype=complex)
    for i in range(1,num_loops+1) :
        grad_list[i] = i*np.power(x_N_2,i-1) / (nt_hooft**((i+2)/2))
        loop[i]=np.sum (np.power(x_N_2,i)) / (nt_hooft**((i+2)/2))
    grad_list [0] = np.zeros(nt_hooft)
    loop[0]=1.0


    #omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range(omega_dim):
        for j in range(omega_dim):
            omega[i,j]=(i+1)*(j+1)*loop[i+j]

    #little_omega=np.zeros ((omega_dim),dtype=complex)
    for i in range(1,omega_dim):
        little_omega[i]=0.0
        for j in range(i):
            little_omega [i] += (i+1)*loop[j]*loop[i-1-j]
    little_omega[0]=0.0
    

    # Solve system of linear equations for LnJ, instead of inverting Omega
    LnJ = np.linalg.solve(omega, little_omega)

    #grad_end= timer()
    #print("Time taken to calculate Omega, little omega and LnJ",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()

    grad_1=np.zeros((nt_hooft),dtype=complex)# dtype=np.float64)
    for i in range (omega_dim):
        for j in range (omega_dim):
            grad_1+= - LnJ[i]*(i+1)*(j+1)*grad_list[i+j]*LnJ[j]
    #grad_end= timer()
    #print("Part 1 of derivative wrt matrix 1 done. Time taken :",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()
    for i in range(1,omega_dim):
        for j in range (i):
            grad_1+=2*LnJ[i]*(i+1)*(grad_list[j]*loop[i-1-j] + loop[j]*grad_list[i-1-j])
    """
    for i in range (omega_dim):
            grad_1+=(-2*LnJ[i]*((i+1)*2*grad_list[i+1]/2.0 + g3_coupling*(i+1)*3*grad_list[i+2]/3.0+g4_coupling*(i+1)*4*grad_list[i+3]))
    """

    grad_1=grad_1/8.0 + grad_list[2]/2.0+g3_coupling*grad_list[3]/3.0+g4_coupling*grad_list[4]

    #grad_end= timer()
    
    return grad_1





#  -------------       Main program    --------------



print ()
lmax = 8 #input ("Enter lmax :    ")
omega_length = int(lmax)

max_size = 2*omega_length-2
omega_size = omega_length
little_omega_size=omega_size-2

print ("Total number of loops is", max_size ,". Omega is a ", omega_size,"x", omega_size,"matrix. " )
print()


N = 50
print ("Number of master variables :", N)
print()

c = 0.0

g_3 = 0.0

g_4 = 1.0


loop=np.zeros ((max_size+1),dtype=complex)
omega=np.zeros ((omega_size,omega_size),dtype=complex)
little_omega=np.zeros ((omega_size),dtype=complex)
LnJ=np.zeros ((omega_size),dtype=complex)

# initial distribution of egenvalues for both matrices
x_init=(np.random.rand(N) - np.ones(N)*0.5)*np.sqrt(N)

print("Starting minimization")
start = timer()

#effect_pot(x_init, omega_size,max_size, little_omega_size, N, c, g_3,g_4)
#effect_pot_grad(x_init, omega_size,max_size, little_omega_size, N, c, g_3,g_4)
#Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='nelder-mead', options={'disp': True, 'maxfev': 1000000, 'maxiter': 1000000})
Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='BFGS', jac=effect_pot_grad, options={'disp': True, 'maxiter': 20000,'gtol': np.sqrt(N)*1e-8})#changed from e-16 20/03/21#fixed N 20/04/2021
#Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='CG', jac=effect_pot_grad, options={'disp': True, 'maxiter': 5000,'gtol': np.sqrt(N*(N+1))*1e-16})

end = timer()
#print((end - start), "sec in minimization")

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

print()
#print ("Energy =", Energy.fun)
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
            A[i,k]+=(i+1)*(k_delta(j,k+1)*loop[i-1-j] + loop[j]*k_delta(i-1-j,k+1))


A_2=np.zeros((omega_size,omega_size,omega_size),dtype=complex)
for k in range (omega_size):
    for l in range (omega_size):
        for i in range(1,omega_size):
            for j in range (i):
                A_2[i,k,l]+=(i+1)*(k_delta(j,k+1)*k_delta(i-1-j,l+1) + k_delta(j,l+1)*k_delta(i-1-j,k+1))


omega_grad=np.zeros((omega_size,omega_size,omega_size),dtype=complex)
for k in range (omega_size):
    for i in range(omega_size):
        for j in range(omega_size):
            omega_grad[i,j,k]+=(i+1)*(j+1)*k_delta(i+j,k+1)


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







"""  -----------        Derivatives with respect to ALL loops -------------    """

loop_gradient=np.zeros((max_size+1,max_size),dtype=int) #note: only up to lmax, not max_size
for i in range (1,max_size+1):
    for j in range (max_size):
        loop_gradient[i,j] = k_delta(i,j+1)

pot_grad_1=np.zeros((max_size),dtype=complex)# dtype=np.float64)
for i in range (omega_size):
    for j in range (omega_size):
        pot_grad_1+= - LnJ[i]*(i+1)*(j+1)*loop_gradient[i+j]*LnJ[j]
for i in range(1,omega_size):
    for j in range (i):
        pot_grad_1+=2*LnJ[i]*(i+1)*(loop_gradient[j]*loop[i-1-j] + loop[j]*loop_gradient[i-1-j])

pot_grad_1=pot_grad_1/8.0 + loop_gradient[2]/2.0+g_3*loop_gradient[3]/3.0+g_4*loop_gradient[4]

print()
print("Smallest derivative of potential wrt loops:",",",np.format_float_scientific(np.amin(np.abs(pot_grad_1)),precision=6))
print("Largest derivative of potential wrt loops:",",",np.format_float_scientific(np.amax(np.abs(pot_grad_1)),precision=6))



A_all=np.zeros((omega_size,max_size),dtype=complex)
for k in range (max_size):
    for i in range(1,omega_size):
        for j in range (i):
            A_all[i,k]+=(i+1)*(k_delta(j,k+1)*loop[i-1-j] + loop[j]*k_delta(i-1-j,k+1))

A_2_all=np.zeros((omega_size,max_size,max_size),dtype=complex)
for k in range (max_size):
    for l in range (max_size):
        for i in range(1,omega_size):
            for j in range (i):
                A_2_all[i,k,l]+=(i+1)*(k_delta(j,k+1)*k_delta(i-1-j,l+1) + k_delta(j,l+1)*k_delta(i-1-j,k+1))

omega_grad_all=np.zeros((omega_size,omega_size,max_size),dtype=complex)
for k in range (max_size):
    for i in range(omega_size):
        for j in range(omega_size):
            omega_grad_all[i,j,k]+=(i+1)*(j+1)*k_delta(i+j,k+1)


omega_deriv_all= np.sum(np.array([[[omega_grad_all[i,j,k]*LnJ[j] for k in range(max_size)]for i in range(omega_size)]for j in range(omega_size)]),axis=0)


V_2_all=multi_dot([A_all.T,omega_inv,A_all])

V_2_all+=np.sum(np.array([[[A_2_all[i,k,l]*LnJ[i] for k in range(max_size)]for l in range(max_size)]for i in range(omega_size)]),axis=0)



V_2_all+=multi_dot([omega_deriv_all.T,omega_inv,omega_deriv_all])

V_2_all-=multi_dot([A_all.T, omega_inv,omega_deriv_all])+multi_dot([omega_deriv_all.T, omega_inv,A_all])




sub_grad_list=grad_list[1:max_size+1,:]
print()
print ("Wrt eigenvalues - mass squared")
#sub_grad_list=grad_list[1:omega_size+1,:]
#print (sub_grad_list.shape)
i=0
for x in list(np.sort(linalg.eigvals(multi_dot([sub_grad_list.T,V_2_all,sub_grad_list])).real)*N*N/4.0):
    if (i==0): print ("{:.8f}". format(x), "\n   .....  ")
    if (i > N-omega_size-2): print ("{:.8f}". format(x))
    i+=1
    #print (np.format_float_scientific(x,precision=6),",")


print()
print ("Spectrum wrt eigenvalues - taken absolute value of mass squared eigenvalues")
i=0
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([sub_grad_list.T,V_2_all,sub_grad_list])).real)))*N/2.0):
    if (i==0): print ("{:.8f}". format(x), "\n   .....  ")
    if (i > N-omega_size-2): print ("{:.8f}". format(x))
    i+=1


matrix_diag_eigenv= np.diag(np.dot(pot_grad_1,grad_2_list[1:max_size+1,:]))

print()
print ("Wrt eigenvalues plus diagonal")
i=0
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([sub_grad_list.T, V_2_all,sub_grad_list])+matrix_diag_eigenv).real)))*N/2.0):
    if (i==0): print ("{:.8f}". format(x), "\n   .....  ")
    if (i > N-omega_size-2): print ("{:.8f}". format(x))
    i+=1


little_omega_grad=np.zeros((omega_size, N), dtype=complex)
for i in range(1,omega_size):
    for j in range (i):
        little_omega_grad[i]+=(i+1)*(grad_list[j]*loop[i-1-j] + loop[j]*grad_list[i-1-j])


print ()
print("Energy:",",",np.format_float_scientific(Energy.fun,precision=6))
print("Energy:",",",format(Energy.fun.real, '.6f'))