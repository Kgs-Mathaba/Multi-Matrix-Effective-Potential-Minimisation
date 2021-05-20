

import numpy as np
from scipy import linalg
#from itertools import combinations
from scipy.optimize import minimize
from numpy.linalg import multi_dot
from timeit import default_timer as timer
import datetime




def foo(arg):
    print('arg = {}'.format(arg))


def k_delta (i_1,i_2):

    k_delta_result=0
    if (i_1==i_2) :  k_delta_result=1

    return k_delta_result


def initialize_loops(omega_dim, num_loops):
    """
    This function intializes loops, omega, little_omega and LnJ with zeros
    
    Argument:
    omega_dim -- size of omega matrix
    num_loops -- number of loops

    Returns:
    loop -- initialized complex array of shape (num_loops,)
    omega --  initialized 2d complex array of shape (omega_dim,omega_dim)
    little_omega -- initialized complex array of shape (omega_dim,)
    LnJ -- initialized complex array of shape (omega_dim,)
    """

    loop=np.zeros ((num_loops+1),dtype=complex)
    omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    little_omega=np.zeros ((omega_dim),dtype=complex)
    LnJ=np.zeros ((omega_dim),dtype=complex)

    return loop, omega, little_omega, LnJ

def effect_pot(x_N_2, omega_dim, num_loops,litte_omega_dim, nt_hooft, c_coupling, g3_coupling,g4_coupling):
    """
    This func




    """

    print ()
    print ("Function call")
    #func_start= timer()
    loop, omega, little_omega, LnJ = initialize_loops(omega_dim, num_loops)
    # Obtain Loops and derivatives from eigenvalues.
    for i in range(1,num_loops+1) :
      if i<omega_dim+1:
        loop[i]=np.sum (np.power(x_N_2,i)) / (nt_hooft**((i+2)/2))
      else:
        loop[i]=0
    loop[0]=1.0

    


    print()
    # Generate Omega matrix from Loops
    #omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range(omega_dim):
        for j in range(omega_dim):
            omega[i,j]=(i+1)*(j+1)*loop[i+j]

    # Generate "little omega" (omeg) from Loops
    #little_omega=np.zeros ((omega_dim),dtype=complex)
    for i in range(1,omega_dim):
        little_omega[i]=0.0
        for j in range(i):
            little_omega [i] += (i+1)*loop[j]*loop[i-1-j]
    little_omega[0]=0.0
    #little_omega+=- omega[:,1]/2.0-g3_coupling*omega[:,2]/3.0-g4_coupling*omega[:,3]

   
    # Solve system of linear equations for LnJ, instead of inverting Omega
    LnJ = np.linalg.solve(omega, little_omega)
    print("LnJ = ", LnJ)
    # Obtain value of collective effective potential
    large_n_energy = np.dot(little_omega , LnJ)/8.0 + loop[2]/2.0+g3_coupling*loop[3]/3.0+g4_coupling*loop[4]

    print ("Large N Energy :")
    #print (np.format_float_scientific(large_n_energy,precision=6))#doesn't print imaginary part
    print (large_n_energy)
    #func_end= timer()
    #print("Time taken",",", datetime.timedelta(seconds = int(func_end-func_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))

    return large_n_energy.real


def effect_pot_grad(x_N_2, omega_dim, num_loops,litte_omega_dim, nt_hooft, c_coupling, g3_coupling, g4_coupling):

   

    print()
    print ("Gradient function call")
    #grad_start= timer()

    loop, omega, little_omega, LnJ = initialize_loops(omega_dim, num_loops)
    # Obtain Loops and derivatives from eigenvalues.
    grad_list=np.zeros((num_loops+1,nt_hooft),dtype=complex)
    for i in range(1,num_loops+1) :
      if i<omega_dim+1:
        grad_list[i] = i*np.power(x_N_2,i-1) / (nt_hooft**((i+2)/2))
        loop[i] = np.sum (np.power(x_N_2,i)) / (nt_hooft**((i+2)/2))
      else:
        grad_list[i] = 0
        loop[i]= 0
    grad_list[0] = np.zeros(nt_hooft)
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
    #little_omega+=- omega[:,1]/2.0-g3_coupling*omega[:,2]/3.0-g4_coupling*omega[:,3]

    # Solve system of linear equations for LnJ, instead of inverting Omega
    LnJ = np.linalg.solve(omega, little_omega)
    print("grad LnJ = ", LnJ)


    grad_1=np.zeros((nt_hooft),dtype=complex)# dtype=np.float64)
    for i in range (omega_dim):
        for j in range (omega_dim):
            grad_1+= - LnJ[i]*(i+1)*(j+1)*grad_list[i+j]*LnJ[j]
    
    for i in range(1,omega_dim):
        for j in range (i):
            grad_1+=2*LnJ[i]*(i+1)*(grad_list[j]*loop[i-1-j] + loop[j]*grad_list[i-1-j])


    large_n_energy = np.dot(little_omega , LnJ)/8.0 + loop[2]/2.0+g3_coupling*loop[3]/3.0+g4_coupling*loop[4]

    print ("Large N Energy :")
    print (large_n_energy)

    grad_1=grad_1/8.0 + grad_list[2]/2.0+g3_coupling*grad_list[3]/3.0+g4_coupling*grad_list[4]

   
    return grad_1