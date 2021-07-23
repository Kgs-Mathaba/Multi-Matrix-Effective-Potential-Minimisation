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
import sys

from Collective2 import compare, cyclic_generator
from Collective2 import initialize_loops





def effect_pot(x_N_2, omega_dim, num_loops, nt_hooft, c_coupling, g3_coupling,g4_coupling):

    loop, omega, little_omega, LnJ = initialize_loops(omega_size, max_size)

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
    #little_omega+=- omega[:,2]/2.0-omega[:,4]/2.0 + c_coupling*omega[:,3]-g3_coupling*omega[:,5]/3.0-g3_coupling*omega[:,8]/3.0-g4_coupling*omega[:,9]-g4_coupling*omega[:,14]

    """  Kramers rule
    det=np.linalg.det(omega)
    c=np.array(omega)
    for j in range(omega_dim):
        c[:,j]=little_omega
        LnJ[j]=np.linalg.det(c)/det
        c[:,j]=omega[:,j]
    """

    LnJ = np.linalg.solve(omega, little_omega)

    large_n_energy = np.dot(little_omega , LnJ)/8.0 + loop[3]/2.0 + loop[5]/2.0 + c_coupling*loop[4] + g3_coupling*loop[6]/3.0 + g3_coupling*loop[9]/3.0 + g4_coupling*loop[10]/4.0 + g4_coupling*loop[15]/4.0

    print ("Large N Energy :")
    #print (np.format_float_scientific(large_n_energy,precision=6))#doesn't print imaginary part
    print (large_n_energy)
    #func_end= timer()
    #print("Time taken",",", datetime.timedelta(seconds = int(func_end-func_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))

    return large_n_energy.real

    

def effect_pot_grad(x_N_2, omega_dim, num_loops, nt_hooft, c_coupling, g3_coupling, g4_coupling):

    loop, omega, little_omega, LnJ = initialize_loops(omega_size, max_size)

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
    #little_omega+=- omega[:,2]/2.0-omega[:,4]/2.0 + c_coupling*omega[:,3]-g3_coupling*omega[:,5]/3.0-g3_coupling*omega[:,8]/3.0-g4_coupling*omega[:,9]-g4_coupling*omega[:,14]

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

#    for i in range (omega_dim):
#        for indx in range(non_zero[i+1,3]):
#            k=nonzero_index[i+1,3,indx]
#            y=nonzero_y[i+1,3,indx]
#            grad_1+=(-2*LnJ[i]*(y/2.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,5]):
#            k=nonzero_index[i+1,5,indx]
#            y=nonzero_y[i+1,5,indx]
#            grad_1+=(-2*LnJ[i]*(y/2.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,4]):
#            k=nonzero_index[i+1,4,indx]
#            y=nonzero_y[i+1,4,indx]
#            grad_1+=(-2*LnJ[i]*(- c_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,6]):
#            k=nonzero_index[i+1,6,indx]
#            y=nonzero_y[i+1,6,indx]
#            grad_1+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,9]):
#            k=nonzero_index[i+1,9,indx]
#            y=nonzero_y[i+1,9,indx]
#            grad_1+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,10]):
#            k=nonzero_index[i+1,10,indx]
#            y=nonzero_y[i+1,10,indx]
#            grad_1+=(-2*LnJ[i]*(g4_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,15]):
#            k=nonzero_index[i+1,15,indx]
#            y=nonzero_y[i+1,15,indx]
#            grad_1+=(-2*LnJ[i]*(g4_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    grad_1 = grad_1/8.0 
    grad_1+= grad_array_1[3]/(np.power(nt_hooft, (len(loop_list[3])+2)/2))  
    grad_1+= grad_array_1[5]/(np.power(nt_hooft, (len(loop_list[5])+2)/2))
    grad_1+= c_coupling *(grad_array_1[4]/ (np.power(nt_hooft, (len(loop_list[4])+2)/2)))
    grad_1+= g3_coupling*(grad_array_1[6]/ (np.power(nt_hooft, (len(loop_list[6])+2)/2)))
    grad_1+= g3_coupling*(grad_array_1[9]/ (np.power(nt_hooft, (len(loop_list[9])+2)/2)))
    grad_1+= g4_coupling*(grad_array_1[10]/(np.power(nt_hooft, (len(loop_list[10])+2)/2)))
    grad_1+= g4_coupling*(grad_array_1[15]/(np.power(nt_hooft, (len(loop_list[15])+2)/2)))
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

#   for i in range (omega_dim):
#       for indx in range(non_zero[i+1,3]):
#            k=nonzero_index[i+1,3,indx]
#            y=nonzero_y[i+1,3,indx]
#            grad_2+=(-2*LnJ[i]*(y/2.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,5]):
#            k=nonzero_index[i+1,5,indx]
#            y=nonzero_y[i+1,5,indx]
#            grad_2+=(-2*LnJ[i]*(y/2.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,4]):
#            k=nonzero_index[i+1,4,indx]
#            y=nonzero_y[i+1,4,indx]
#            grad_2+=(-2*LnJ[i]*(- c_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,6]):
#            k=nonzero_index[i+1,6,indx]
#            y=nonzero_y[i+1,6,indx]
#            grad_2+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,9]):
#            k=nonzero_index[i+1,9,indx]
#            y=nonzero_y[i+1,9,indx]
#            grad_2+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,10]):
#            k=nonzero_index[i+1,10,indx]
#            y=nonzero_y[i+1,10,indx]
#            grad_2+=(-2*LnJ[i]*(g4_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
#        for indx in range(non_zero[i+1,15]):
#            k=nonzero_index[i+1,15,indx]
#            y=nonzero_y[i+1,15,indx]
#            grad_2+=(-2*LnJ[i]*(g4_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    grad_2 = grad_2/8.0
    grad_2 += grad_list_2[3]/(np.power(nt_hooft, (len(loop_list[3])+2)/2))  
    grad_2 += grad_list_2[5]/(np.power(nt_hooft, (len(loop_list[5])+2)/2))
    grad_2 += c_coupling *(grad_list_2[4]/ (np.power(nt_hooft, (len(loop_list[4])+2)/2)))
    grad_2 += g3_coupling*(grad_list_2[6]/ (np.power(nt_hooft, (len(loop_list[6])+2)/2)))
    grad_2 += g3_coupling*(grad_list_2[9]/ (np.power(nt_hooft, (len(loop_list[9])+2)/2)))
    grad_2 += g4_coupling*(grad_list_2[10]/(np.power(nt_hooft, (len(loop_list[10])+2)/2)))
    grad_2 += g4_coupling*(grad_list_2[15]/(np.power(nt_hooft, (len(loop_list[15])+2)/2)))

    #grad_end= timer()
    #print("Remaining time taken to differentiate wrt matrix 2: ",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")
    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    return np.concatenate((grad_1.real,grad_2.real),axis=0)

"""
  -------------       Main program    --------------
"""


print ()
lmax = sys.argv[1] if len(sys.argv) > 1 else  input ("Enter lmax :    ")
omega_length = int(lmax)
file_name = "lmax_"+str(omega_length)+"_collective_info_v3.npy"
print ()
print("Importing list of loops, Omega and little omega coefficients from file :",file_name  )


import os
import pathlib

PATH = pathlib.Path(__file__).parent.absolute() #get the parent directory of the script
current_folder = str(PATH)
data_folder_name = "data"
data_folder = os.path.join(current_folder, data_folder_name)
data_file_name = "lmax_"+str(omega_length)+"_collective_info_v2.npy"
data_file = os.path.join(data_folder, data_file_name)

from Collective2 import load_npy_data

with open(data_file, 'rb') as f:
    omega_length=np.load(f)
    max_length=np.load(f)
    numb_tuples_list=np.load(f,allow_pickle=True)
    loop_list=np.load(f,allow_pickle=True)
    non_zero=np.load(f,allow_pickle=True)
    nonzero_index=np.load(f,allow_pickle=True)
    nonzero_y=np.load(f,allow_pickle=True)
    non_zero_lo=np.load(f,allow_pickle=True)
    nonzero_loop1_index=np.load(f,allow_pickle=True)
    nonzero_loop2_index=np.load(f,allow_pickle=True)
    nonzero_z=np.load(f,allow_pickle=True)
    adjoint_loops = np.load(f, allow_pickle=True)


max_size = numb_tuples_list [max_length][1]
omega_size = numb_tuples_list [omega_length][1]
little_omega_size = numb_tuples_list [omega_length-2][1]

print ("File import completed. Total number of loops is", max_size ,". Omega is a ", omega_size,"x", omega_size,"matrix. " )
print()

NString =  sys.argv[2] if len(sys.argv) > 2 else input ("Enter 't Hooft N :    ")
N = int(NString)
print ("Number of master variables :", N*(N+1))
print()

cstring = sys.argv[3] if len(sys.argv) > 3 else input ("Enter a value for the quadratic mixing coupling c :   ")
c = float(cstring)
g_3_string = sys.argv[4] if len(sys.argv) > 4 else input ("Enter a value for the cubic coupling g_3 :   ")
g_3 = float(g_3_string)
g_4_string = sys.argv[5] if len(sys.argv) > 5 else  input ("Enter a value for the quartic coupling g_4 :   ")
g_4 = float(g_4_string)







# initial distribution of egenvalues for both matrices
eigen_init=(np.random.rand(N) - np.ones(N)*0.5)*np.sqrt(N)
m_2_init=(np.random.rand(N,N)-np.ones((N,N))*0.5)*np.sqrt(N)

x_init=np.concatenate ((eigen_init, m_2_init.flatten()),axis=0)

print("Starting minimization")
start = timer()
ARGS = (omega_size,max_size, N, c, g_3,g_4)

#effect_pot(x_init, omega_size,max_size, little_omega_size, N, c, g_3,g_4)
#effect_pot_grad(x_init, omega_size,max_size, little_omega_size, N, c, g_3,g_4)
#Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='nelder-mead', options={'disp': True, 'maxfev': 1000000, 'maxiter': 1000000})
#Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='BFGS', jac=effect_pot_grad, options={'disp': True, 'maxiter': 5000,'gtol': np.sqrt(N*(N+1))*1e-16})#changed from e-16 17/02/21
Energy = minimize(effect_pot, x_init, args=ARGS, method='CG', jac=effect_pot_grad, options={'disp': True, 'maxiter': 5000,'gtol': np.sqrt(N*(N+1))*1e-16})

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
loop, omega, little_omega, LnJ = initialize_loops(omega_size, max_size)

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
#print("Accuracy of SD equations:")
#print ("Most accurate SD equation accuracy:",",", np.format_float_scientific(np.amin(np.abs(little_omega)),precision=6))
#print ("Least accurate SD equation accuracy:", ",",np.format_float_scientific(np.amax(np.abs(little_omega)),precision=6))
print("Energy:",",",np.format_float_scientific(Energy.fun,precision=6))

#print ("Energy =", Energy.fun)
for i in range (max_size+1):
    print ("Loop,", i,",=,",loop_list[i], ",has value,", format(loop[i].real, '.6f'), ",imaginary part,", format(loop[i].imag, '.6f'))#round (loop[i].real,5)

#omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
for i in range (omega_size):
    for j in range (omega_size):
        omega[i,j]=0.0
        for indx in range(non_zero[i+1,j+1]):
            k=nonzero_index[i+1,j+1,indx]
            y=nonzero_y[i+1,j+1,indx]
            omega[i,j]+= y * loop[k]