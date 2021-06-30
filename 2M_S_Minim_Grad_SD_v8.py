#attempt to get spectrum from master variables
#   ?? this code loads _collective_info_v3 files
# changed normalization of Gradient


import numpy as np
from scipy import linalg
from itertools import combinations
from scipy.optimize import minimize
from numpy.linalg import multi_dot
from timeit import default_timer as timer
import datetime

def compare (Loop1, Loop2):

  """
  Compare two loops up to cyclic permutations
  """
  Length1 = len (Loop1)
  Length2 = len (Loop2)

  Same=False
  if (Length2 == 0) :
          Same=True
          return Same

  #Create warning if loops passed with different lengths #print ("lengths", Length1, Length2)

  Same = False
  for i in range (Length2):
    if np.array_equal (Loop1, np.roll(Loop2, i)):
        Same=True
        break

  return Same

def cyclic_generator (lp):

    """

    Input:   lp is a (1,len(lp)) ndarray
    Output:  is an array containing the first of the lexicographically sorted set of cyclic permutations of lp
             - example: [2,1,2,1] --> [1,2,1,2]
    """


    cyclic_perm = np.array([[0]*len(lp)]*len(lp))

    for i in range (len(lp)):
      cyclic_perm[i]= np.roll(lp, i)

    first_lexi_array = np.array (sorted([tuple(row) for row in cyclic_perm]))[0:1, :]

    return first_lexi_array



def effect_pot(x_N_2, omega_dim, num_loops,litte_omega_dim, nt_hooft, c_coupling, g3_coupling,g4_coupling):

    global loop
    global omega
    global little_omega
    global LnJ

    print ()
    print ("Function call")
    #func_start= timer()

    matrices_array =np.zeros((2,nt_hooft,nt_hooft),dtype=complex)
    matrices_array[0] = np.diag(x_N_2[0:nt_hooft])
    temp_matrix_2 = np.reshape (x_N_2[nt_hooft:nt_hooft**2+nt_hooft], (nt_hooft,nt_hooft))
    real_matrix_2 = np.triu(temp_matrix_2)
    real_matrix_2 = real_matrix_2+real_matrix_2.T-np.diag(np.diag(temp_matrix_2))
    imag_matrix_2 = np.tril(temp_matrix_2)
    imag_matrix_2 = imag_matrix_2-imag_matrix_2.T
    matrices_array[1]=real_matrix_2+imag_matrix_2*1.j

    for i in range (1,num_loops+1):
        loop_matrix=matrices_array[loop_list[i][0]-1]
        for k in range(1,len(loop_list[i])):
            if loop_list[i][k]==1 :
                loop_matrix=loop_matrix * x_N_2[0:nt_hooft]          #[k]#matrices_list[loop_list[i][k]-1]
            else:
                loop_matrix=np.dot(loop_matrix,matrices_array[1])#[loop_list[i][k]-1]
        loop[i]=np.trace(loop_matrix)/( np.power(nt_hooft, (len(loop_list[i])+2)/2)) #v3
    loop[0]=1.0

    #omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range (omega_dim):
        for j in range (omega_dim):
            omega[i,j]=0.0
            for indx in range(non_zero[i+1,j+1]):
                k=nonzero_index[i+1,j+1,indx]
                y=nonzero_y[i+1,j+1,indx]
                omega[i,j]+= y * loop[k]

    #little_omega=np.zeros ((omega_dim),dtype=complex)
    for i in range (omega_dim):
        little_omega[i]=0.0
        for indx in range(non_zero_lo[i+1]):
            j=nonzero_loop1_index[i+1,indx]
            k=nonzero_loop2_index[i+1,indx]
            z=nonzero_z[i+1,indx]
            little_omega[i]+= z*loop[j]*loop[k]
    little_omega+=- omega[:,2]/2.0-omega[:,4]/2.0 + c_coupling*omega[:,3]-g3_coupling*omega[:,5]/3.0-g3_coupling*omega[:,8]/3.0-g4_coupling*omega[:,9]-g4_coupling*omega[:,14]

    """  Kramers rule
    det=np.linalg.det(omega)
    c=np.array(omega)
    for j in range(omega_dim):
        c[:,j]=little_omega
        LnJ[j]=np.linalg.det(c)/det
        c[:,j]=omega[:,j]
    """

    LnJ = np.linalg.solve(omega, little_omega)

    large_n_energy = np.dot(little_omega , LnJ)/8.0

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
    #global grad_array_1
    #global grad_list_2

    print()
    print ("Gradient function call")
    #grad_start= timer()

    matrices_array =np.zeros((2,nt_hooft,nt_hooft),dtype=complex)
    matrices_array[0] = np.diag(x_N_2[0:nt_hooft])
    temp_matrix_2 = np.reshape (x_N_2[nt_hooft:nt_hooft**2+nt_hooft], (nt_hooft,nt_hooft))
    real_matrix_2 = np.triu(temp_matrix_2)
    real_matrix_2 = real_matrix_2+real_matrix_2.T-np.diag(np.diag(temp_matrix_2))
    imag_matrix_2 = np.tril(temp_matrix_2)
    imag_matrix_2 = imag_matrix_2-imag_matrix_2.T
    matrices_array[1]=real_matrix_2+imag_matrix_2*1.j

    # Loops calculated with derivatives
    loop[0]=1.0
    loop[1]=np.sum(x_N_2[0:nt_hooft]) /  ( np.power(nt_hooft, 1.5))
    loop[2]=np.trace(matrices_array[1])/( np.power(nt_hooft, 1.5))

    grad_array_1=np.zeros((num_loops+1,nt_hooft),dtype=complex)##
    grad_array_2=np.zeros((num_loops+1,nt_hooft,nt_hooft),dtype=complex)
    grad_list_2=np.zeros((num_loops+1,nt_hooft*nt_hooft),dtype=complex)
    grad_array_1[1]= np.ones(nt_hooft)  # this assumes first 3 loops are [] [1] [2]
    grad_array_2[2]= np.identity(nt_hooft)
    grad_list_2[2]=(grad_array_2[2]).flatten()

    for i in range (3,num_loops+1):    #assumes first 3 loops will alwyas be the same
        for k in range(len(loop_list[i])):
            loop_left = loop_list[i][0:k]
            loop_right= loop_list[i][k+1:len(loop_list[i])]
            loop_grad=np.concatenate ((loop_right , loop_left))
            grad_len = len(loop_grad)
            grad_matrix=matrices_array[loop_grad[0]-1]

            if grad_len > 1 :
                for l in range(1,grad_len):
                    if loop_grad[l]==1 :
                        grad_matrix=grad_matrix * x_N_2[0:nt_hooft]          #[k]#matrices_list[loop_list[i][k]-1]
                    else:
                        grad_matrix=np.dot(grad_matrix,matrices_array[1])
            if loop_list[i][k]==1 :
                grad_array_1[i]+=np.diagonal(grad_matrix.real)
            else :
                grad_array_2[i]+=grad_matrix

        last_index=len(loop_list[i])-1
        if loop_list[i][last_index]==1:
            loop_matrix=grad_matrix* x_N_2[0:nt_hooft]
        else:
            loop_matrix=np.dot(grad_matrix,matrices_array[1])#[

        loop[i]=np.trace(loop_matrix)/( np.power(nt_hooft, (len(loop_list[i])+2)/2))
        upper_matrix_2 = np.triu((grad_array_2[i] + grad_array_2[i].T))- np.diag(np.diag(grad_array_2[i]))
        lower_matrix_2 = -1.0j*(np.tril((grad_array_2[i] - grad_array_2[i].T)))
        grad_array_2[i] = (upper_matrix_2+lower_matrix_2)
        grad_list_2[i]=(grad_array_2[i]).flatten()

    #grad_end= timer()
    #print("Time taken to compute derivatives",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()


    #omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range (omega_dim):
        for j in range (omega_dim):
            omega[i,j]=0.0
            for indx in range(non_zero[i+1,j+1]):
                k=nonzero_index[i+1,j+1,indx]
                y=nonzero_y[i+1,j+1,indx]
                omega[i,j]+= y * loop[k]

    #little_omega=np.zeros ((omega_dim),dtype=complex)
    for i in range (omega_dim):
        little_omega[i]=0.0
        for indx in range(non_zero_lo[i+1]):
            j=nonzero_loop1_index[i+1,indx]
            k=nonzero_loop2_index[i+1,indx]
            z=nonzero_z[i+1,indx]
            little_omega[i]+= z*loop[j]*loop[k]
    little_omega+=- omega[:,2]/2.0-omega[:,4]/2.0 + c_coupling*omega[:,3]-g3_coupling*omega[:,5]/3.0-g3_coupling*omega[:,8]/3.0-g4_coupling*omega[:,9]-g4_coupling*omega[:,14]

    """  Kramers rule
    det=np.linalg.det(omega)
    c=np.array(omega)
    for j in range(omega_dim):
        c[:,j]=little_omega
        LnJ[j]=np.linalg.det(c)/det
        c[:,j]=omega[:,j]
    """

    LnJ = np.linalg.solve(omega, little_omega)

    #grad_end= timer()
    #print("Time taken to calculate Omega, little omega and LnJ",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()

    grad_1=np.zeros((nt_hooft),dtype=complex)# dtype=np.float64)

    for i in range (omega_dim):
        for j in range (omega_dim):
            for indx in range(non_zero[i+1,j+1]):
                k=nonzero_index[i+1,j+1,indx]
                y=nonzero_y[i+1,j+1,indx]
                grad_1+= - LnJ[i]*y*LnJ[j]* grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2))

    #grad_end= timer()
    #print("Part 1 of derivative wrt matrix 1 done. Time taken :",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()

    for i in range (omega_dim):
        for indx in range(non_zero_lo[i+1]):
            j=nonzero_loop1_index[i+1,indx]
            k=nonzero_loop2_index[i+1,indx]
            z=nonzero_z[i+1,indx]
            grad_1+=2*LnJ[i]*z*(grad_array_1[j]*loop[k]/(np.power(nt_hooft,(len(loop_list[j])+2)/2))+ loop[j]*grad_array_1[k]/(np.power(nt_hooft, (len(loop_list[k])+2)/2)))

    for i in range (omega_dim):
        for indx in range(non_zero[i+1,3]):
            k=nonzero_index[i+1,3,indx]
            y=nonzero_y[i+1,3,indx]
            grad_1+=(-2*LnJ[i]*(y/2.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,5]):
            k=nonzero_index[i+1,5,indx]
            y=nonzero_y[i+1,5,indx]
            grad_1+=(-2*LnJ[i]*(y/2.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,4]):
            k=nonzero_index[i+1,4,indx]
            y=nonzero_y[i+1,4,indx]
            grad_1+=(-2*LnJ[i]*(- c_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,6]):
            k=nonzero_index[i+1,6,indx]
            y=nonzero_y[i+1,6,indx]
            grad_1+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,9]):
            k=nonzero_index[i+1,9,indx]
            y=nonzero_y[i+1,9,indx]
            grad_1+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,10]):
            k=nonzero_index[i+1,10,indx]
            y=nonzero_y[i+1,10,indx]
            grad_1+=(-2*LnJ[i]*(g4_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,15]):
            k=nonzero_index[i+1,15,indx]
            y=nonzero_y[i+1,15,indx]
            grad_1+=(-2*LnJ[i]*(g4_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    grad_1=grad_1/8.0

    #grad_end= timer()
    #print("Derivative wrt matrix 1 done. Remaining time taken :",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()

    grad_2=np.zeros((nt_hooft*nt_hooft),dtype=complex)

    for i in range (omega_dim):
        for j in range (omega_dim):
            for indx in range(non_zero[i+1,j+1]):
                k=nonzero_index[i+1,j+1,indx]
                y=nonzero_y[i+1,j+1,indx]
                grad_2+= - LnJ[i]*y*LnJ[j]* grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2))

    #grad_end= timer()
    #print("Part 1 of derivative wrt matrix 2 done. Time taken :",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()

    for i in range (omega_dim):
        for indx in range(non_zero_lo[i+1]):
            j=nonzero_loop1_index[i+1,indx]
            k=nonzero_loop2_index[i+1,indx]
            z=nonzero_z[i+1,indx]
            grad_2+=2*LnJ[i]*z*(grad_list_2[j]*loop[k]/(np.power(nt_hooft,(len(loop_list[j])+2)/2))+ loop[j]*grad_list_2[k]/(np.power(nt_hooft, (len(loop_list[k])+2)/2)))

    for i in range (omega_dim):
        for indx in range(non_zero[i+1,3]):
            k=nonzero_index[i+1,3,indx]
            y=nonzero_y[i+1,3,indx]
            grad_2+=(-2*LnJ[i]*(y/2.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,5]):
            k=nonzero_index[i+1,5,indx]
            y=nonzero_y[i+1,5,indx]
            grad_2+=(-2*LnJ[i]*(y/2.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,4]):
            k=nonzero_index[i+1,4,indx]
            y=nonzero_y[i+1,4,indx]
            grad_2+=(-2*LnJ[i]*(- c_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,6]):
            k=nonzero_index[i+1,6,indx]
            y=nonzero_y[i+1,6,indx]
            grad_2+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,9]):
            k=nonzero_index[i+1,9,indx]
            y=nonzero_y[i+1,9,indx]
            grad_2+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,10]):
            k=nonzero_index[i+1,10,indx]
            y=nonzero_y[i+1,10,indx]
            grad_2+=(-2*LnJ[i]*(g4_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
        for indx in range(non_zero[i+1,15]):
            k=nonzero_index[i+1,15,indx]
            y=nonzero_y[i+1,15,indx]
            grad_2+=(-2*LnJ[i]*(g4_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    grad_2=grad_2/8.0

    #grad_end= timer()
    #print("Remaining time taken to differentiate wrt matrix 2: ",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")
    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    return np.concatenate((grad_1.real,grad_2.real),axis=0)

"""
  -------------       Main program    --------------
"""


print ()
lmax = input ("Enter lmax :    ")
omega_length = int(lmax)
file_name = "lmax_"+str(omega_length)+"_collective_info_v3.npy"
print ()
print("Importing list of loops, Omega and little omega coefficients from file :",file_name  )

with open(file_name, 'rb') as f:
    omega_length=np.load(f)
    max_length=np.load(f)
    numb_tuples_list=np.load(f,allow_pickle=True)
    loop_list=np.load(f,allow_pickle=True)
    non_zero=np.load(f,allow_pickle=True)
    nonzero_index=np.load(f,allow_pickle=True)
    nonzero_y=np.load(f,allow_pickle=True)
    old_y=np.load(f,allow_pickle=True)
    non_zero_lo=np.load(f,allow_pickle=True)
    nonzero_loop1_index=np.load(f,allow_pickle=True)
    nonzero_loop2_index=np.load(f,allow_pickle=True)
    nonzero_z=np.load(f,allow_pickle=True)
    old_z = np.load(f,allow_pickle=True)

max_size = numb_tuples_list [max_length][1]
omega_size = numb_tuples_list [omega_length][1]
little_omega_size = numb_tuples_list [omega_length-2][1]

print ("File import completed. Total number of loops is", max_size ,". Omega is a ", omega_size,"x", omega_size,"matrix. " )
print()

NString = input ("Enter 't Hooft N :    ")
N = int(NString)
print ("Number of master variables :", N*(N+1))
print()

cstring = input ("Enter a value for the quadratic mixing coupling c :   ")
c = float(cstring)
g_3_string = input ("Enter a value for the cubic coupling g_3 :   ")
g_3 = float(g_3_string)
g_4_string = input ("Enter a value for the quartic coupling g_4 :   ")
g_4 = float(g_4_string)


print ()
print ("Identifying conjugate loops")

start_adjoint = timer()
adjoint_loops=np.zeros ((max_size+1),dtype=int)
adjoint_loops[0]=0
for i in range (1,max_size+1):
    adjoint =  cyclic_generator(np.flip(loop_list[i]))
    temp_adjoint = adjoint.T
    temp_len = np.ma.size(temp_adjoint, axis=0)
    adjoint = np.reshape(temp_adjoint,(temp_len,))

    if all (adjoint ==loop_list[i]):
        adjoint_loops[i]=i
    else:
        start=numb_tuples_list [temp_len][0]
        end=numb_tuples_list [temp_len][1]
        for p in range(start,end+1):
            if compare (loop_list[p],adjoint): adjoint_loops[i]=p
    """
    if not (all (adjoint ==loop_list[i])) :
        print ("Adjoint of loop ", i , loop_list[i], " up to cyclic is ", adjoint,". This is loop ", adjoint_loops[i])
    """

end_adjoint = timer()
print("Done. Time taken :", (end_adjoint - start_adjoint))
print ()


"""
adjoint_omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
for i in range (omega_dim):
    for j in range (omega_dim):
        adjoint_omega[i,j]=0.0
        for k in range (num_loops+1):
            adjoint_omega[i,j]+=y[adjoint_loops[i+1],j+1,k] * loop[k]
"""

loop=np.zeros ((max_size+1),dtype=complex)
omega=np.zeros ((omega_size,omega_size),dtype=complex)
little_omega=np.zeros ((omega_size),dtype=complex)
LnJ=np.zeros ((omega_size),dtype=complex)

# initial distribution of egenvalues for both matrices
eigen_init=(np.random.rand(N) - np.ones(N)*0.5)*np.sqrt(N)
m_2_init=(np.random.rand(N,N)-np.ones((N,N))*0.5)*np.sqrt(N)

x_init=np.concatenate ((eigen_init, m_2_init.flatten()),axis=0)

print("Starting minimization")
start = timer()

#effect_pot(x_init, omega_size,max_size, little_omega_size, N, c, g_3,g_4)
#effect_pot_grad(x_init, omega_size,max_size, little_omega_size, N, c, g_3,g_4)
#Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='nelder-mead', options={'disp': True, 'maxfev': 1000000, 'maxiter': 1000000})
#Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='BFGS', jac=effect_pot_grad, options={'disp': True, 'maxiter': 5000,'gtol': np.sqrt(N*(N+1))*1e-16})#changed from e-16 17/02/21
Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='CG', jac=effect_pot_grad, options={'disp': True, 'maxiter': 5000,'gtol': np.sqrt(N*(N+1))*1e-16})

end = timer()
#print((end - start), "sec in minimization")


matrix_1_final = np.diag(Energy.x[0:N])
temp_matrix_2_final = np.reshape (Energy.x[N:N**2+N], (N,N))
real_matrix_2_final = np.triu(temp_matrix_2_final)
real_matrix_2_final = real_matrix_2_final+real_matrix_2_final.T-np.diag(np.diag(temp_matrix_2_final))
imag_matrix_2_final = np.tril(temp_matrix_2_final)
imag_matrix_2_final = imag_matrix_2_final-imag_matrix_2_final.T
matrix_2_final=real_matrix_2_final+imag_matrix_2_final*1.j


matrices_array_final =np.zeros((2,N,N),dtype=complex)
matrices_array_final[0] =matrix_1_final
matrices_array_final[1] =matrix_2_final

"""
for i in range (1,max_size+1):
    loop_matrix=matrices_array_final[loop_list[i][0]-1]
    for k in range(1,len(loop_list[i])):
        loop_matrix=np.dot(loop_matrix,matrices_array_final[loop_list[i][k]-1])
    loop[i]=np.trace(loop_matrix) / ( np.power(N, (len(loop_list[i])+2)/2)) #note: not declared as complex; always returns real value
loop[0]=1.
"""

""" from gradient function """


# Loops calculated with derivatives
loop[0]=1.0
loop[1]=np.sum(Energy.x[0:N]) /  ( np.power(N, 1.5))
loop[2]=np.trace(matrices_array_final[1])/( np.power(N, 1.5))

grad_array_1=np.zeros((max_size+1,N),dtype=complex)##
grad_array_2=np.zeros((max_size+1,N,N),dtype=complex)
grad_list_2=np.zeros((max_size+1,N*N),dtype=complex)
grad_array_1[1]= np.ones(N) / ( np.power(N, 1.5)) # this assumes first 3 loops are [] [1] [2]
grad_array_2[2]= np.identity(N)/( np.power(N, 1.5))
grad_list_2[2]=(grad_array_2[2]).flatten()

for i in range (3,max_size+1):    #assumes first 3 loops will alwyas be the same
    for k in range(len(loop_list[i])):
        loop_left = loop_list[i][0:k]
        loop_right= loop_list[i][k+1:len(loop_list[i])]
        loop_grad=np.concatenate ((loop_right , loop_left))
        grad_len = len(loop_grad)
        grad_matrix=matrices_array_final[loop_grad[0]-1]

        if grad_len > 1 :
            for l in range(1,grad_len):
                if loop_grad[l]==1 :
                    grad_matrix=grad_matrix * Energy.x[0:N]          #[k]#matrices_list[loop_list[i][k]-1]
                else:
                    grad_matrix=np.dot(grad_matrix,matrices_array_final[1])
        if loop_list[i][k]==1 :
            grad_array_1[i]+=np.diagonal(grad_matrix.real)
        else :
            grad_array_2[i]+=grad_matrix

    last_index=len(loop_list[i])-1
    if loop_list[i][last_index]==1:
        loop_matrix=grad_matrix* Energy.x[0:N]
    else:
        loop_matrix=np.dot(grad_matrix,matrices_array_final[1])#[

    loop[i]=np.trace(loop_matrix)/( np.power(N, (len(loop_list[i])+2)/2))
    upper_matrix_2 = np.triu((grad_array_2[i] + grad_array_2[i].T))- np.diag(np.diag(grad_array_2[i]))
    lower_matrix_2 = -1.0j*(np.tril((grad_array_2[i] - grad_array_2[i].T)))
    grad_array_2[i] = (upper_matrix_2+lower_matrix_2)/( np.power(N, (len(loop_list[i])+2)/2))
    grad_list_2[i]=(grad_array_2[i]).flatten()
    grad_array_1[i]=grad_array_1[i]/( np.power(N, (len(loop_list[i])+2)/2))



""" ----------------------"""

print ()
print ("c =",c,",", "g_3 =", g_3, ",","g_4=", g_4 )
print ("No of loops =",max_size, ",","lmax =",omega_length, ",","Omega is", omega_size,"x", omega_size, "matrix")
print ( "N =", N, ",","No of master variables = ", N*(N+1))
#np.format_float_scientific(np.amin(np.abs(np.concatenate((grad_1.real,grad_2.real),axis=0))),precision=6)

print("Minimization time",",", datetime.timedelta(seconds = int(end-start)), "(days,h:m:s)")

print("Smallest gradient component modulus:",",",np.format_float_scientific(np.amin(np.abs(Energy.jac)),precision=6))
print("Largest gradient component modulus:",",",np.format_float_scientific(np.amax(np.abs(Energy.jac)),precision=6))
print("Accuracy of SD equations:")
print ("Most accurate SD equation accuracy:",",", np.format_float_scientific(np.amin(np.abs(little_omega)),precision=6))
print ("Least accurate SD equation accuracy:", ",",np.format_float_scientific(np.amax(np.abs(little_omega)),precision=6))
print("Energy:",",",np.format_float_scientific(Energy.fun,precision=6))

#print ("Energy =", Energy.fun)
for i in range (max_size+1):
    print ("Loop,", i,",=,",loop_list[i], ",has value,", format(loop[i].real, '.6f'), ",imaginary part,", format(loop[i].imag, '.6f'))#round (loop[i].real,5)

print()
for i in range (omega_size):
    print ("SD,",loop_list[i+1],",i,", i+1,",=,",little_omega[i])#round (loop[i].real,5)

"""
omega_2=omega[:,:]

print()
print ("Last Omega eigenvalues")
print (np.sort(linalg.eigvals(omega_2).real))
"""

#omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
for i in range (omega_size):
    for j in range (omega_size):
        omega[i,j]=0.0
        for indx in range(non_zero[i+1,j+1]):
            k=nonzero_index[i+1,j+1,indx]
            y=nonzero_y[i+1,j+1,indx]
            omega[i,j]+= y * loop[k]

print()
print ("Explicitly Symmetric Omega to accuracy 1e-15 ? " ,",", np.all ((np.isclose(omega.T, omega,atol=1e-15 ))))
print ("Explicitly Symmetric Omega to accuracy 1e-16 ? " ,",", np.all ((np.isclose(omega.T, omega,atol=1e-16 ))))
print ("Explicitly Symmetric Omega to accuracy 1e-17 ? " ,",", np.all ((np.isclose(omega.T, omega,atol=1e-17 ))))
print()

print ("Symmetric Omega eigenvalues")
Eigen = np.sort(linalg.eigvals(omega))
i=0
for x in list(np.sort(linalg.eigvals(omega))):
    print ( np.format_float_scientific(Eigen[i].real ,precision=6),",", np.format_float_scientific(Eigen[i].imag ,precision=6),",",  )
    i+=1


print()

print ("Symmetric Omega eigenvalues - real part only")
for x in list(np.sort(linalg.eigvals(omega).real)):
    print (np.format_float_scientific(x,precision=6),",")
#print (np.sort(linalg.eigvals(omega).real))
print()

"""
h_omega=np.zeros ((omega_size,omega_size),dtype=complex)
for i in range (omega_size):
    for j in range (omega_size):
        h_omega[i,j]=0.0
        for k in range (max_size+1):
            h_omega[i,j]+=y[i+1,adjoint_loops[j+1],k] * loop[k]
"""

h_omega=np.zeros ((omega_size,omega_size),dtype=complex)
for i in range (omega_size):
    for j in range (omega_size):
        h_omega[i,j]=0.0
        for indx in range(non_zero[i+1,adjoint_loops[j+1]]):
            k=nonzero_index[i+1,adjoint_loops[j+1],indx]
            y=nonzero_y[i+1,adjoint_loops[j+1],indx]
            h_omega[i,j]+= y * loop[k]

print()
print ("Hermitian Omega eigenvalues")
Eigen = np.sort(linalg.eigvals(h_omega))
i=0
for x in list(np.sort(linalg.eigvals(h_omega))):
    print ( np.format_float_scientific(Eigen[i].real ,precision=6),",", np.format_float_scientific(Eigen[i].imag ,precision=6),",",  )
    i+=1

print()

for i in range (omega_size):
    for j in range (omega_size):
        h_omega[i,j]=omega[i,adjoint_loops[j+1]-1]

print ("h_omega hermitian ? ", np.array_equal (np.conj(h_omega), h_omega.T))

print()
print ("Another Hermitian Omega eigenvalues - check - real part only")
for x in list(np.sort(linalg.eigvals(h_omega).real)):
    print (np.format_float_scientific(x,precision=6),",")
#print (np.sort(linalg.eigvals(omega).real))
print()



"""
print()
print ("Reconstructed Omega eigenvalues")
print (np.sort(linalg.eigvals(omega).real))
print()
"""

A_old=np.zeros((omega_size,omega_size),dtype=complex)
for i in range (omega_size):
    for j in range (little_omega_size):
        for k in range (little_omega_size+1):
        #A[i,j]-=np.sum (np.array([z[i+1,j+1,k]*loop[k] for k in range (little_omega_size+1)]))    ##axis?
        #A[i,j]-=np.sum (np.array([z[i+1,k,j+1]*loop[k] for k in range (little_omega_size+1)]))
            A_old[i,j]+=old_z[i+1,j+1,k]*loop[k]    ##axis?
            A_old[i,j]+=old_z[i+1,k,j+1]*loop[k]
    for j in range (omega_size):
        A_old[i,j]-=(old_y[i+1,3,j+1]/2.0 +old_y[i+1,5,j+1]/2.0 - c*old_y[i+1,4,j+1]+g_3*old_y[i+1,6,j+1]/3.0+g_3*old_y[i+1,9,j+1]/3.0+g_4*old_y[i+1,10,j+1]+g_4*old_y[i+1,15,j+1])


A=np.zeros((omega_size,omega_size),dtype=complex)
for i in range (omega_size):
    for indx in range(non_zero_lo[i+1]):
        j=nonzero_loop1_index[i+1,indx]
        k=nonzero_loop2_index[i+1,indx]
        z=nonzero_z[i+1,indx]
        if j != 0 : A[i,j-1]+=z*loop[k]    ##axis?
        if k != 0 : A[i,k-1]+=z*loop[j]
    for indx in range(non_zero[i+1,3]):
        k=nonzero_index[i+1,3,indx]
        y=nonzero_y[i+1,3,indx]
        if k in range(1,omega_size+1): A[i,k-1]-= y/2.0
    for indx in range(non_zero[i+1,5]):
        k=nonzero_index[i+1,5,indx]
        y=nonzero_y[i+1,5,indx]
        if k in range(1,omega_size+1): A[i,k-1]-= y/2.0
    for indx in range(non_zero[i+1,4]):
        k=nonzero_index[i+1,4,indx]
        y=nonzero_y[i+1,4,indx]
        if k in range(1,omega_size+1): A[i,k-1]+= c*y
    for indx in range(non_zero[i+1,6]):
        k=nonzero_index[i+1,6,indx]
        y=nonzero_y[i+1,6,indx]
        if k in range(1,omega_size+1): A[i,k-1]-= g_3*y/3.0
    for indx in range(non_zero[i+1,9]):
        k=nonzero_index[i+1,9,indx]
        y=nonzero_y[i+1,9,indx]
        if k in range(1,omega_size+1): A[i,k-1]-= g_3*y/3.0
    for indx in range(non_zero[i+1,10]):
        k=nonzero_index[i+1,10,indx]
        y=nonzero_y[i+1,10,indx]
        if k in range(1,omega_size+1): A[i,k-1]-= g_4*y
    for indx in range(non_zero[i+1,15]):
        k=nonzero_index[i+1,15,indx]
        y=nonzero_y[i+1,15,indx]
        if k in range(1,omega_size+1): A[i,k-1]-= g_4*y


print ("A matrices the same ? ", np.array_equal (A_old, A))


omega_inv=linalg.inv(omega)

"""
print ("c =",c,",", "g_3 =", g_3, ",","g_4=", g_4 )
print ("No of loops =",max_size, ",","lmax =",omega_length, ",","Omega is", omega_size,"x", omega_size, "matrix")
print ( "N =", N, ",","No of master variables = ", N*(N+1))

print()
print ("Omega eigenvalues")
print (np.sort(linalg.eigvals(omega).real))
"""
print()
print ("Loop based fluctuations spectrum")
#print (np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])))))
#print (np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])).real))/2.0)
for x in list(np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])).real))/2.0):
    print (np.format_float_scientific(x,precision=6),",")
print ()
print ("fixed no. of decimal points")

for x in list(np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])).real))/2.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))

h_left_omega=np.zeros((omega_size, omega_size), dtype=complex)
for i in range (omega_size):
    for j in range (omega_size):
        h_left_omega[i,j]=omega[adjoint_loops[i+1]-1,j]

print()
print ("h_left_omega Hermitian ?", np.array_equal (np.conj(h_left_omega), h_left_omega.T))

print()
print("Hermitean loop spectrum ?")

m_2 = multi_dot([h_left_omega, A.T, omega_inv, A])

print ("m_2 hermitian ? ", np.array_equal (np.conj(m_2), m_2.T))

for x in list(np.sort(np.sqrt(linalg.eigvals(m_2)).real)/2.0):
    print (np.format_float_scientific(x,precision=6),",")
print ()
print ("fixed no. of decimal points")

for x in list(np.sort(np.sqrt(linalg.eigvals(m_2)).real)/2.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))


print()
print ("Submatrices specturm")
print()

if g_4==0.0:
    lmax_sub_min=3
else:
    lmax_sub_min=4

lmax_sub_max = (omega_length+2)//2
if lmax_sub_max >= lmax_sub_min :
    for i in range (lmax_sub_min,lmax_sub_max+1):
        sub_matrix_size = numb_tuples_list [i][1]
        sub_A = A[0:sub_matrix_size,0:sub_matrix_size]
        sub_omega = omega[0:sub_matrix_size,0:sub_matrix_size]
        sub_omega_inv = linalg.inv(sub_omega)
        print ("Submatrix with lmax = ", i ,"has size" , sub_matrix_size , "x", sub_matrix_size )

        sub_spectrum_matrix = multi_dot([sub_omega, sub_A.T, sub_omega_inv, sub_A])
        #print (sub_A)
        for x in list(np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0):
            print (np.format_float_scientific(x,precision=6),",")
        print()
        print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        print()
        print ("second version")
        sub_A=np.zeros((sub_matrix_size,sub_matrix_size),dtype=complex)
        sub_little_omega_size = numb_tuples_list [i-2][1]
        for l in range (sub_matrix_size):
            for j in range (sub_little_omega_size):
                for k in range (sub_little_omega_size+1):
                #A[i,j]-=np.sum (np.array([z[i+1,j+1,k]*loop[k] for k in range (little_omega_size+1)]))    ##axis?
                #A[i,j]-=np.sum (np.array([z[i+1,k,j+1]*loop[k] for k in range (little_omega_size+1)]))
                    sub_A[l,j]+=old_z[l+1,j+1,k]*loop[k]    ##axis?
                    sub_A[l,j]+=old_z[l+1,k,j+1]*loop[k]
            for j in range (sub_matrix_size):
                sub_A[l,j]-=(old_y[l+1,3,j+1]/2.0 +old_y[l+1,5,j+1]/2.0 - c*old_y[l+1,4,j+1]+g_3*old_y[l+1,6,j+1]/3.0+g_3*old_y[l+1,9,j+1]/3.0+g_4*old_y[l+1,10,j+1]+g_4*old_y[l+1,15,j+1])
        sub_spectrum_matrix = multi_dot([sub_omega, sub_A.T, sub_omega_inv, sub_A])
        #print (sub_A)
        """
        Eigen = np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix)))/2.0
        i=0
        for x in list(np.sort(linalg.eigvals(sub_spectrum_matrix))):
            print ( np.format_float_scientific(Eigen[i].real ,precision=6),",", np.format_float_scientific(Eigen[i].imag ,precision=6),",",  )
            i+=1

        #print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        """
        for x in list(np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0):
            #print (np.format_float_scientific(x,precision=6),",")
            print ("{:.8f}". format(x))
        print()
        print()
        print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        print()

print ("Obtain A using non zero entries " )

if lmax_sub_max >= lmax_sub_min :
    for i in range (lmax_sub_min,lmax_sub_max+1):
        sub_matrix_size = numb_tuples_list [i][1]
        sub_A = A[0:sub_matrix_size,0:sub_matrix_size]
        sub_omega = omega[0:sub_matrix_size,0:sub_matrix_size]
        sub_omega_inv = linalg.inv(sub_omega)
        print ("Submatrix with lmax = ", i ,"has size" , sub_matrix_size , "x", sub_matrix_size )

        sub_spectrum_matrix = multi_dot([sub_omega, sub_A.T, sub_omega_inv, sub_A])
        #print (sub_A)
        for x in list(np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0):
            print (np.format_float_scientific(x,precision=6),",")
        print()
        print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        print()
        print ("second version")
        sub_A=np.zeros((sub_matrix_size,sub_matrix_size),dtype=complex)
        sub_little_omega_size = numb_tuples_list [i-2][1]
        for l in range (sub_matrix_size):
            for indx in range(non_zero_lo[l+1]):
                j=nonzero_loop1_index[l+1,indx]
                k=nonzero_loop2_index[l+1,indx]
                z=nonzero_z[l+1,indx]
                if j != 0 : sub_A[l,j-1]+=z*loop[k]    ##axis?
                if k != 0 : sub_A[l,k-1]+=z*loop[j]
            for indx in range(non_zero[l+1,3]):
                k=nonzero_index[l+1,3,indx]
                y=nonzero_y[l+1,3,indx]
                if k in range(1,sub_matrix_size+1): sub_A[l,k-1]-= y/2.0
            for indx in range(non_zero[l+1,5]):
                k=nonzero_index[l+1,5,indx]
                y=nonzero_y[l+1,5,indx]
                if k in range(1,sub_matrix_size+1): sub_A[l,k-1]-= y/2.0
            for indx in range(non_zero[l+1,4]):
                k=nonzero_index[l+1,4,indx]
                y=nonzero_y[l+1,4,indx]
                if k in range(1,sub_matrix_size+1): sub_A[l,k-1]+= c*y
            for indx in range(non_zero[l+1,6]):
                k=nonzero_index[l+1,6,indx]
                y=nonzero_y[l+1,6,indx]
                if k in range(1,sub_matrix_size+1): sub_A[l,k-1]-= g_3*y/3.0
            for indx in range(non_zero[l+1,9]):
                k=nonzero_index[l+1,9,indx]
                y=nonzero_y[l+1,9,indx]
                if k in range(1,sub_matrix_size+1): sub_A[l,k-1]-= g_3*y/3.0
            for indx in range(non_zero[l+1,10]):
                k=nonzero_index[l+1,10,indx]
                y=nonzero_y[l+1,10,indx]
                if k in range(1,sub_matrix_size+1): sub_A[l,k-1]-= g_4*y
            for indx in range(non_zero[l+1,15]):
                k=nonzero_index[l+1,15,indx]
                y=nonzero_y[l+1,15,indx]
                if k in range(1,sub_matrix_size+1): sub_A[l,k-1]-= g_4*y
        sub_spectrum_matrix = multi_dot([sub_omega, sub_A.T, sub_omega_inv, sub_A])
        """
        Eigen = np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix)))/2.0
        i=0
        for x in list(np.sort(linalg.eigvals(sub_spectrum_matrix))):
            print ( np.format_float_scientific(Eigen[i].real ,precision=6),",", np.format_float_scientific(Eigen[i].imag ,precision=6),",",  )
            i+=1

        #print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        """
        for x in list(np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0):
            #print (np.format_float_scientific(x,precision=6),",")
            print ("{:.8f}". format(x))
        print()

        print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        print()



"""
print ("Master Field")

for y in Energy.x :
    print (y)



print (Energy.x)
"""

"""     ---------------------------------------------------------------------------
Master variabels spectrum

"""
""" &&&&&&&&&   from gradient function """

""" &&&&&&&&& """
A_master=np.zeros((omega_size,max_size),dtype=complex)
for i in range (omega_size):
    for indx in range(non_zero_lo[i+1]):
        j=nonzero_loop1_index[i+1,indx]
        k=nonzero_loop2_index[i+1,indx]
        z=nonzero_z[i+1,indx]
        if j != 0 : A_master[i,j-1]+=z*loop[k]    ##axis?
        if k != 0 : A_master[i,k-1]+=z*loop[j]
    for indx in range(non_zero[i+1,3]):
        k=nonzero_index[i+1,3,indx]
        y=nonzero_y[i+1,3,indx]
        if k in range(1,max_size+1): A_master[i,k-1]-= y/2.0
    for indx in range(non_zero[i+1,5]):
        k=nonzero_index[i+1,5,indx]
        y=nonzero_y[i+1,5,indx]
        if k in range(1,max_size+1): A_master[i,k-1]-= y/2.0
    for indx in range(non_zero[i+1,4]):
        k=nonzero_index[i+1,4,indx]
        y=nonzero_y[i+1,4,indx]
        if k in range(1,max_size+1): A_master[i,k-1]+= c*y
    for indx in range(non_zero[i+1,6]):
        k=nonzero_index[i+1,6,indx]
        y=nonzero_y[i+1,6,indx]
        if k in range(1,max_size+1): A_master[i,k-1]-= g_3*y/3.0
    for indx in range(non_zero[i+1,9]):
        k=nonzero_index[i+1,9,indx]
        y=nonzero_y[i+1,9,indx]
        if k in range(1,max_size+1): A_master[i,k-1]-= g_3*y/3.0
    for indx in range(non_zero[i+1,10]):
        k=nonzero_index[i+1,10,indx]
        y=nonzero_y[i+1,10,indx]
        if k in range(1,max_size+1): A_master[i,k-1]-= g_4*y
    for indx in range(non_zero[i+1,15]):
        k=nonzero_index[i+1,15,indx]
        y=nonzero_y[i+1,15,indx]
        if k in range(1,max_size+1): A_master[i,k-1]-= g_4*y

print()
print ("Master variables spectrum")
print()
print ("Shape of A_master", A_master.shape)
print ("Shape of grad_array_1", grad_array_1.shape)
print ("Shape of grad_list_2", grad_list_2.shape)

grad_master=np.zeros ((max_size,N*(N+1)), dtype=complex)
for i in range(1,max_size+1):
    grad_master[i-1]= np.concatenate((grad_array_1[i],grad_list_2[i]),axis=0)
    #print("checking concatenation")
    #print (grad_array_1[i].real,grad_list_2[i].real,grad_master[i-1].real )
print ("Shape of grad_master", grad_master.shape)

print()
print ("Wrt eigenvalues - mass squared")
#sub_grad_list=grad_list[1:omega_size+1,:]
#print (sub_grad_list.shape)
i=0
for x in list(np.sort(linalg.eigvals(multi_dot([grad_master.T,A_master.T, omega_inv, A_master,grad_master])).real)*N*N/4.0):
    if (i==0): print ("{:.8f}". format(x), "\n   .....  ")
    if (i > N*(N+1)-omega_size-2): print ("{:.8f}". format(x))
    i+=1
    #print (np.format_float_scientific(x,precision=6),",")


print()
print ("Spectrum wrt eigenvalues - taken absolute value of mass squared eigenvalues")
i=0
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([grad_master.T,A_master.T, omega_inv, A_master,grad_master])).real)))*N/2.0):
    if (i==0): print ("{:.8f}". format(x), "\n   .....  ")
    if (i > N*(N+1)-omega_size-2): print ("{:.8f}". format(x))
    i+=1
    #print (np.format_float_scientific(x,precision=6),",")


print()
print("Some checks - unfilled omega")

grad_sub_master = grad_master[0:omega_size]
print ("shape of grad_sub_master", grad_sub_master.shape)

omega_master=multi_dot([grad_sub_master,grad_sub_master.T])
print ()
print ("shape of omega_master",omega_master.shape )

print()
print ("Omega and Omega_master the same? ", np.array_equal (omega, omega_master))


print()
print ("Same to accuracy 1e-15 ? " ,",", np.all ((np.isclose(omega, omega_master,atol=1e-08))))
print ("Same to accuracy 1e-16 ? " ,",", np.all ((np.isclose(omega, omega_master,atol=1e-09 ))))
print ("Same to accuracy 1e-17 ? " ,",", np.all ((np.isclose(omega, omega_master,atol=1e-10 ))))


"""
print()
print("Omega")
print(omega)

print()
print("Omega_master")
print(omega_master)
"""


print()
print ("eigenvalues - mass squared - master variables restricted to omega")
#sub_grad_list=grad_list[1:omega_size+1,:]
#print (sub_grad_list.shape)
i=0
for x in list(np.sort(linalg.eigvals(multi_dot([grad_sub_master.T,A.T, omega_inv, A,grad_sub_master])).real)*N*N/4.0):
    if (i==0): print ("{:.8f}". format(x), "\n   .....  ")
    if (i > N*(N+1)-omega_size-2): print ("{:.8f}". format(x))
    i+=1
    #print (np.format_float_scientific(x,precision=6),",")


print()
print ("Spectrum wrt eigenvalues - taken absolute value of mass squared eigenvalues")
i=0
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([grad_sub_master.T,A.T, omega_inv, A,grad_sub_master])).real)))*N/2.0):
    if (i==0): print ("{:.8f}". format(x), "\n   .....  ")
    if (i > N*(N+1)-omega_size-2): print ("{:.8f}". format(x))
    i+=1




"""
RunNString = input ("Enter run number (integer):    ")
RunN = int(RunNString)



file_name = "lmax="+str(omega_length)+"_N="+str(N)+"_c="+ str(c)+"_g3="+ str(g_3)+"_g4="+ str(g_4)+"_Run"+str(RunN)+"_Master_Field.npy"



with open(file_name, 'wb') as f:
    np.save(f,c)
    np.save(f,g_3)
    np.save(f,g_4)
    np.save(f,omega_length)
    np.save(f,N)
    np.save (f, Energy.x)

with open(file_name, 'rb') as f:
    loaded_c=np.load(f)
    loaded_g_3=np.load(f)
    loaded_g_4=np.load(f)
    loaded_omega_length=np.load(f)
    loaded_N=np.load(f)
    loaded_master_field = np.load(f)

"""





"""
print ("Info extracted from file")
print (loaded_c)
print (loaded_g_3)
print (loaded_g_4)
print (loaded_omega_length)
print (loaded_N)
print ("Extracted Master Field")
print (loaded_master_field)


for y in loaded_master_field :
    print (y)



"""


#print ("eigenvalues of the hessian")

#for i in range ( len(linalg.eigvals(linalg.inv(Energy.hess_inv)))):
#    print (linalg.eigvals(linalg.inv(Energy.hess_inv))[i])
