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
        loop[i]=np.sum (np.power(x_N_2,i)) / (nt_hooft**((i+2)/2))
    loop[0]=1.0

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
    little_omega+=- omega[:,1]/2.0-g3_coupling*omega[:,2]/3.0-g4_coupling*omega[:,3]

    """  Kramers rule
    det=np.linalg.det(omega)
    c=np.array(omega)
    for j in range(omega_dim):
        c[:,j]=little_omega
        LnJ[j]=np.linalg.det(c)/det
        c[:,j]=omega[:,j]
    """
    # Solve system of linear equations for LnJ, instead of inverting Omega
    LnJ = np.linalg.solve(omega, little_omega)

    # Obtain value of collective effective potential
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
    little_omega+=- omega[:,1]/2.0-g3_coupling*omega[:,2]/3.0-g4_coupling*omega[:,3]

    """  Kramers rule
    det=np.linalg.det(omega)
    c=np.array(omega)
    for j in range(omega_dim):
        c[:,j]=little_omega
        LnJ[j]=np.linalg.det(c)/det
        c[:,j]=omega[:,j]
    """
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

    for i in range (omega_dim):
            grad_1+=(-2*LnJ[i]*((i+1)*2*grad_list[i+1]/2.0 + g3_coupling*(i+1)*3*grad_list[i+2]/3.0+g4_coupling*(i+1)*4*grad_list[i+3]))

    grad_1=grad_1/8.0

    #grad_end= timer()
    #print("Derivative wrt matrix 1 done. Remaining time taken :",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #grad_start= timer()

    #grad_end= timer()
    #print("Remaining time taken to differentiate wrt matrix 2: ",",", datetime.timedelta(seconds = int(grad_end-grad_start)), "(days,h:m:s)")
    #print ("Concatenated derivative", np.around(np.concatenate((grad_1.real,grad_2.real),axis=0),decimals=6))
    #print()
    #print ("Gradient")
    #print (grad_1)
    return grad_1





#  -------------       Main program    --------------



print ()
lmax = input ("Enter lmax :    ")
omega_length = int(lmax)

max_size = 2*omega_length-2
omega_size = omega_length
little_omega_size=omega_size-2

print ("Total number of loops is", max_size ,". Omega is a ", omega_size,"x", omega_size,"matrix. " )
print()

NString = input ("Enter 't Hooft N :    ")
N = int(NString)
print ("Number of master variables :", N)
print()

c = 0.0
g_3_string = input ("Enter a value for the cubic coupling g_3 :   ")
g_3 = float(g_3_string)
g_4_string = input ("Enter a value for the quartic coupling g_4 :   ")
g_4 = float(g_4_string)


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
Energy = minimize (effect_pot, x_init, args=(omega_size,max_size, little_omega_size, N, c, g_3,g_4), method='BFGS', jac=effect_pot_grad, options={'disp': True, 'maxiter': 5000,'gtol': np.sqrt(N*(N+1))*1e-14})#changed from e-16 20/03/21
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
print ("c =",c,",", "g_3 =", g_3, ",","g_4=", g_4 )
print ("No of loops =",max_size, ",","lmax =",omega_length, ",","Omega is", omega_size,"x", omega_size, "matrix")
print ( "No of master variables = ", N)
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
    print ("Loop,", i, ",has value,", format(loop[i].real, '.6f'), ",imaginary part,", format(loop[i].imag, '.6f'))#round (loop[i].real,5)

print()
for i in range (omega_size):
    print ("SD,", i+1,",=,",little_omega[i])#round (loop[i].real,5)



"""
omega_2=omega[:,:]

print()
print ("Last Omega eigenvalues")
print (np.sort(linalg.eigvals(omega_2).real))
"""

#omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
for i in range(omega_size):
    for j in range(omega_size):
        omega[i,j]=(i+1)*(j+1)*loop[i+j]

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

A=np.zeros((omega_size,omega_size),dtype=complex)
for i in range (2,omega_size):
    for j in range (i-1):
        A[i,j]= 2*(i+1)*loop[i-j-2]    ##axis?
for i in range (omega_size):
    A[i,i]-=2*(i+1)/2.0
    if (i < omega_size-1) : A[i,i+1]-=3*(i+1)*g_3/3.0
    if (i < omega_size-2) : A[i,i+2]-=4*(i+1)*g_4

A_all=np.zeros((omega_size,max_size),dtype=complex)
for k in range (max_size):
    for i in range(1,omega_size):
        for j in range (i):
            A_all[i,k]+=(i+1)*(k_delta(j,k+1)*loop[i-1-j] + loop[j]*k_delta(i-1-j,k+1))
for i in range (omega_size):
    A_all[i,i]-=2*(i+1)/2.0
    if (i < max_size-1) : A_all[i,i+1]-=3*(i+1)*g_3/3.0
    if (i < max_size-2) : A_all[i,i+2]-=4*(i+1)*g_4




omega_inv=linalg.inv(omega)


pot_grad_loops=np.zeros((omega_size),dtype=complex)
for i in range (omega_size-1):
    for j in range (i):
        pot_grad_loops[i]+= - LnJ[j]*(j+1)*(i+2-j)*LnJ[i+1-j]
for j in range (1,omega_size):
        print ("j", j)
        pot_grad_loops[omega_size-1]+= - LnJ[j]*(j+1)*(omega_size+1-j)*LnJ[omega_size-j]
for i in range (omega_size):
    for j in range (omega_size):
        pot_grad_loops[i]+= - A[j,i]*LnJ[j]

pot_grad_loops=pot_grad_loops/8.0

print()
for i in range (omega_size):
    print ("Gradient wrt loops", i+1,",=,",pot_grad_loops[i])#round (loop[i].real,5)


"""
print ("c =",c,",", "g_3 =", g_3, ",","g_4=", g_4 )
print ("No of loops =",max_size, ",","lmax =",omega_length, ",","Omega is", omega_size,"x", omega_size, "matrix")
print ( "N =", N, ",","No of master variables = ", N*(N+1))

print()
print ("Omega eigenvalues")
print (np.sort(linalg.eigvals(omega).real))
"""
print()
print ("Fluctuations spectrum")
#print (np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])))))
#print (np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])).real))/2.0)
for x in list(np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])).real))/2.0):
    print (np.format_float_scientific(x,precision=6),",")
print ()
print ("fixed no. of decimal points")

for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])).real)))/2.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))

print()
print ("Fluctuations spectrum - 2")
#print (np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])))))
#print (np.sort(np.sqrt(linalg.eigvals(multi_dot([omega, A.T, omega_inv, A])).real))/2.0)
for x in list(np.sort(np.sqrt(linalg.eigvals(multi_dot([omega_inv, A, omega, A.T])).real))/2.0):
    print (np.format_float_scientific(x,precision=6),",")
print ()
print ("fixed no. of decimal points")

for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([omega_inv, A, omega, A.T])).real)))/2.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))








print()
print ("Hessian")
for x in list(np.sort(np.sqrt(linalg.eigvals(linalg.inv(Energy.hess_inv).real))/2.0)):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))

print()
print ("Wrt squared eigenvalues")
sub_grad_list=grad_list[1:omega_size+1,:]
#print (sub_grad_list.shape)
for x in list(np.sort(linalg.eigvals(multi_dot([sub_grad_list.T, A.T, omega_inv, A,sub_grad_list])).real)/4.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))

print()
print ("Wrt eigenvalues")
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([sub_grad_list.T, A.T, omega_inv, A,sub_grad_list])).real)))*N/2.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))


little_omega_grad=np.zeros((omega_size, N), dtype=complex)
for i in range(1,omega_size):
    for j in range (i):
        little_omega_grad[i]+=(i+1)*(grad_list[j]*loop[i-1-j] + loop[j]*grad_list[i-1-j])
for i in range (omega_size):
    little_omega_grad[i]-=(i+1)*2*grad_list[i+1]/2.0
    if (i < omega_size-1) : little_omega_grad[i]-=g_3*(i+1)*3*grad_list[i+2]/3.0
    if (i < omega_size-2) : little_omega_grad[i]-=g_4*(i+1)*4*grad_list[i+3]



#    + g_3*(i+1)*3*grad_list[i+2]/3.0+g_4*(i+1)*4*grad_list[i+3]))
print ("Diagonal matrix in spectrum")
matrix_diag_eigenv= np.diag(np.dot(pot_grad_loops,grad_2_list[1:omega_size+1,:]))
print ("Shape", matrix_diag_eigenv.shape)

print("Smallest entry  modulus:",",",np.format_float_scientific(np.amin(np.abs(matrix_diag_eigenv)),precision=6))
print("Largest entry   modulus:",",",np.format_float_scientific(np.amax(np.abs(matrix_diag_eigenv)),precision=6))



print()
print ("Wrt eigenvalues plus diagonal")
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([sub_grad_list.T, A.T, omega_inv, A,sub_grad_list])+matrix_diag_eigenv).real)))*N/2.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))

print()
print ("Wrt squared eigenvalues - other")
#sub_grad_list=grad_list[0:omega_size,:]
#print (sub_grad_list.shape)
for x in list(np.sort(linalg.eigvals(multi_dot([little_omega_grad.T, omega_inv, little_omega_grad])).real)*N*N/4.0):
    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))

print()
print ("Wrt eigenvalues - other")
#sub_grad_list=grad_list[0:omega_size,:]
#print (sub_grad_list.shape)
for x in list(np.sort(np.sqrt(np.absolute(linalg.eigvals(multi_dot([little_omega_grad.T, omega_inv, little_omega_grad])).real)))*N/2.0):

    #print (np.format_float_scientific(x,precision=6),",")
    print ("{:.8f}". format(x))



print()
print ("With respect to all loops - only first term of V2")

sub_grad_list=grad_list[1:max_size+1,:]
V_2_all=multi_dot([A_all.T,omega_inv,A_all])

print()
print ("Wrt eigenvalues - mass squared - all loops")
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



"""
print ("Submatrices spectrum")
print()

if g_4==0.0:
    lmax_sub_min=3
else:
    lmax_sub_min=4

lmax_sub_max = (omega_length+2)//2
if lmax_sub_max >= lmax_sub_min :
    for i in range (lmax_sub_min,lmax_sub_max+1):
        sub_matrix_size = i
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
        for l in range (2,sub_matrix_size):
                for j in range (l-1):
                    sub_A[l,j]= 2*(l+1)*loop[l-j-2]    ##axis?
        for l in range (sub_matrix_size):
            sub_A[l,l]-=2*(l+1)/2.0
            if (l < sub_matrix_size-1) : sub_A[l,l+1]-=3*(l+1)*g_3/3.0
            if (l < sub_matrix_size-2) : sub_A[l,l+2]-=4*(l+1)*g_4

        sub_spectrum_matrix = multi_dot([sub_omega, sub_A.T, sub_omega_inv, sub_A])
        #print (sub_A)

        Eigen = np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix)))/2.0
        i=0
        for x in list(np.sort(linalg.eigvals(sub_spectrum_matrix))):
            print ( np.format_float_scientific(Eigen[i].real ,precision=6),",", np.format_float_scientific(Eigen[i].imag ,precision=6),",",  )
            i+=1

        #print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        
        for x in list(np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0):
            #print (np.format_float_scientific(x,precision=6),",")
            print ("{:.8f}". format(x))
        print()
        print()
        print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        print()

        for x in list(np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0):
            #print (np.format_float_scientific(x,precision=6),",")
            print ("{:.8f}". format(x))
        print()

        print (np.sort(np.sqrt(linalg.eigvals(sub_spectrum_matrix).real))/2.0)
        print()

"""

"""
print ("Master Field")

for y in Energy.x :
    print (y)



print (Energy.x)


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
