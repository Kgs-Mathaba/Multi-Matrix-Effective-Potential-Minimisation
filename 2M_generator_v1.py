import numpy as np
from scipy import linalg
#from itertools import permutations
from itertools import combinations
#from itertools import combinations_with_replacement
from scipy.optimize import minimize
from scipy.special import binom
from numpy.linalg import multi_dot
from timeit import default_timer as timer

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

##################################################################################
    # old code
    # create an ordered set (no repeted elements) and store it back as an array
    # CyclicPermSet = np.array (sorted({tuple(row) for row in CyclicPerm}))
    # Fill in copies so that number of rows is len(Lp)
    # Rows= np.ma.size(CyclicPermSet, axis=0)
    # CyclicPermSet = np.vstack (([CyclicPermSet]*(len(Lp)/Rows)))
    # print ('Shape of CyclicPermSet=', np.shape(CyclicPermSet))
    # FlatCyclicArray = CyclicPermSet.flatten()[0:len(Lp)]###
############################################################################


def loops_of_length (loop_length):

    """
    Generates all distinct loops (up to cyclic permutations) of length loop_length
    Returns: Stack (matrix) of (1,loop_length) loop arrays

    """

    print("Starting with loops of length ", loop_length)

    """ Version -- 1  """

    all_ones = np.array ([1]*loop_length)
    big_big_perm_array=all_ones

    for i in range (1,loop_length):

        print ("      Starting with", i, "[2]s" )
        comb = list(combinations (range(loop_length),i))
        numb_combinations=len(comb)
        big_perm_array = np.array([[0]*(loop_length)]*numb_combinations)
        for p in range(numb_combinations):
            perm_distinct=list (all_ones)
            for comb_index in comb[p]:
                perm_distinct[comb_index]= 2
            big_perm_array[p]=cyclic_generator(np.array(perm_distinct))
        big_perm_array = np.array (sorted({tuple(row) for row in big_perm_array}))
        big_big_perm_array = np.vstack((big_big_perm_array,big_perm_array))

    big_big_perm_array = np.vstack((big_big_perm_array,np.array ([2]*loop_length)))

    #big_big_perm_set = np.array (sorted({tuple(row) for row in big_big_perm_array}))

    return big_big_perm_array


"""
Main program
"""

#global loop_list
#global numb_tuples_list
#global y
#global z


print ()
lmax = input ("Enter lmax :    ")
omega_length = int(lmax)
#omega_length = 6
max_length = 2*omega_length-2

loop_list =[]                     # loop_list is a list of arrays, as arrays (loops) have different lengths
loop_list.append(np.array([]))

numb_tuples_list =[]
start=0
end=0
numb_tuples_list.append((start,end))
print ()
print ("Generating loops ...")
start1=timer()
for i in range (1,max_length+1):
    start=end+1
    loops_fixed_length = loops_of_length (i)
    num_fixed_length=np.ma.size(loops_fixed_length, axis=0)
    for j in range(num_fixed_length):
        loop_list.append(loops_fixed_length [j:j+1 ,:])
    end=start+num_fixed_length-1
    numb_tuples_list.append((start,end))
end1=timer()


#  Change (1,n) shape to (n,) for easier loop manipulation
for i in range(len(loop_list)):
    temp_loop = loop_list[i].T
    temp_len = np.ma.size(temp_loop, axis=0)
    loop_list[i] = np.reshape(temp_loop,(temp_len,))


#print (numb_tuples_list)
print ("Loops generated. Time taken : ", (end1-start1), " seconds")
print ()

max_size = numb_tuples_list [max_length][1]
print ("Total number of loops up to length ", max_length, " : ", max_size)
print ("List of loops :")
i=0
for x in loop_list:
    print ("Loop", i,"=", x)
    i+=1

print()
for i in range (max_length+1):
    num=numb_tuples_list[i][1]-numb_tuples_list[i][0]+1
    print ("Number of loops of length " ,i, " : ", num,)

print()



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
print (" Here is adjoint_loops: ", adjoint_loops)
#print("dtype of adjoint_loops: ", type(adjoint_loops))


print("\n")



print ("Joining loops and generating Omega matrix ")
omega_size = numb_tuples_list [omega_length][1]
print ("Omega is a ", omega_size, "x",omega_size, " matrix" )
max_size = numb_tuples_list [max_length][1]

y=np.zeros((omega_size+1, omega_size+1, max_size+1),dtype=int)




for i in range (1,omega_size+1):
    print ("      Starting row ", i)
    for j in range (i,omega_size+1):
        loop_1 = loop_list [i]
        loop_2 = loop_list [j]
        #print ("loop_1", loop_1, "loop_2", loop_2    )
        for k in range (len(loop_1)):
            for l in range (len(loop_2)):
                if loop_1[k]==loop_2[l]:
                    loop_1_left = loop_1[0:k]
                    #print (loop_1_left)
                    loop_1_right= loop_1[k+1:len(loop_1)]
                    #print (loop_1_right)
                    loop_2_left = loop_2[0:l]
                    #print (loop_2_left)
                    loop_2_right= loop_2[l+1:len(loop_2)]
                    #print (loop_2_right)
                    #joined_loop = np.concatenate ((loop_1_right , loop_2_left,loop_2_right ,loop_1_left ))
                    joined_loop = np.concatenate ((loop_1_right , loop_1_left,loop_2_right ,loop_2_left ))
                    #print (str(loop_1), str(loop_2), (joined_loop))
                    start=numb_tuples_list [len(joined_loop)][0]
                    #print(start)
                    end=numb_tuples_list [len(joined_loop)][1]
                    #print (end)
                    for p in range(start,end+1):
                        if compare (loop_list[p],joined_loop):
                            #print ("identified loop")
                            y[i][j][p]+=1
                            if i != j : y[j][i][p]+=1

                            #print (y[i][j][p])




"""
for i in range (1,omega_size+1):
    for j in range (1,omega_size+1):
        for k in range (max_size+1):
            if (y[i][j][k])!= 0. :
                print ("Loop", i,"=",loop_list[i],"joins with", "Loop", j,"=", loop_list[j], "into Loop", k,"=", loop_list[k], (y[i][j][k]), "times." )
"""

non_zero=np.zeros((omega_size+1, omega_size+1),dtype=int)
for i in range (1,omega_size+1):
    for j in range (1,omega_size+1):
        non_zero[i,j]=0
        for k in range (max_size+1):
            if (y[i][j][k])!= 0. :
                non_zero[i,j]+=1

max_nonzero_size = np.amax(non_zero)

print()
print("Maximum number of loops that are joined from any given two loops :")
print(max_nonzero_size)


nonzero_index=np.zeros((omega_size+1, omega_size+1, max_nonzero_size),dtype=int)
nonzero_y=np.zeros((omega_size+1, omega_size+1, max_nonzero_size),dtype=int)
for i in range (1,omega_size+1):
    for j in range (1,omega_size+1):
        indx=0
        for k in range (max_size+1):
            if (y[i][j][k])!= 0. :
                nonzero_index[i,j,indx]=k
                nonzero_y[i,j,indx]=y[i,j,k]
                indx+=1

print ()
print("Do you want to print how loops join (this can be a long list) ?")
yes_or_no = "w"
while not (yes_or_no=="y" or yes_or_no=="n"):
    yes_or_no=input ("Enter 'y' or 'n' :    ")



if yes_or_no=="y":
    print()
    for i in range (1,omega_size+1):
        for j in range (1,omega_size+1):
        #print ("range", non_zero[i,j])
            for indx in range(non_zero[i,j]):
                print ("Loop", i,"=",loop_list[i],"joins with", "Loop", j,"=", loop_list[j], "into Loop", nonzero_index[i,j,indx],"=", loop_list[nonzero_index[i,j,indx]], nonzero_y[i,j,indx], "times." )







#should check symmetry of y
# Start little omega

print()
print ("Splitting loops and generating little omega ")
"""
print ("Recall loops:")

i=0
for x in loop_list:
    print ("Loop", i,"=", x)
    i+=1
"""

#print ("Little omega")
little_omega_size = numb_tuples_list [omega_length-2][1]
print ("Little omega splits into loops up to loop number ", little_omega_size)
#z=np.zeros ((omega_size+1,little_omega_size+1, little_omega_size+1))
z=np.zeros ((omega_size+1,omega_size+1, omega_size+1),dtype=int)

for i in range (3,omega_size+1):
    for k in range (len(loop_list[i])):
        for l in range (len(loop_list[i])):
            if k!=l :
                if loop_list[i][k]== loop_list[i][l]:
                    if k<l :
                        loop_inside=loop_list[i][k+1:l]
                        #print (i,loop_list[i], k,l, loop_inside)
                        #loop_outside_1 =
                        loop_outside=np.concatenate((loop_list[i][l+1:len(loop_list[i])],loop_list[i][0:k]))
                        #print (i,loop_list[i], k,l, loop_inside, loop_outside)
                        start=numb_tuples_list [len(loop_inside)][0]
                        end=numb_tuples_list [len(loop_inside)][1]
                        for p in range(start,end+1):
                            if compare (loop_list[p],loop_inside): loop_inside_index=p
                        start=numb_tuples_list [len(loop_outside)][0]
                        end=numb_tuples_list [len(loop_outside)][1]
                        for p in range(start,end+1):
                            if compare (loop_list[p],loop_outside):loop_outside_index=p
                        #z[i][loop_inside_index][loop_outside_index]+=1
                    else:
                        loop_inside=loop_inside=loop_list[i][l+1:k]
                        loop_outside=np.concatenate((loop_list[i][k+1:len(loop_list[i])],loop_list[i][0:l] ) )
                        #print (i,loop_list[i],k,l, loop_inside, loop_outside)
                        start=numb_tuples_list [len(loop_inside)][0]
                        end=numb_tuples_list [len(loop_inside)][1]
                        for p in range(start,end+1):
                            if compare (loop_list[p],loop_inside): loop_inside_index=p
                        start=numb_tuples_list [len(loop_outside)][0]
                        end=numb_tuples_list [len(loop_outside)][1]
                        for p in range(start,end+1):
                            if compare (loop_list[p],loop_outside):loop_outside_index=p
                    z[i][loop_inside_index][loop_outside_index]+=1




# turn z into upper diagonal
for i in range (1,omega_size+1):
    for j in range (little_omega_size+1):
        for k in range (j+1,little_omega_size+1):
            z[i,j,k] = z[i,j,k]+ z[i,k,j]
            z[i,k,j] = 0
"""
for i in range (1,omega_size+1):
    for j in range (little_omega_size+1):
        for k in range (little_omega_size+1):
            if (z[i][j][k])!= 0. :
                print ("Loop", i,"=",loop_list[i],"breaks into", "Loop", j,"=", loop_list[j], "and Loop", k,"=", loop_list[k], (z[i][j][k]), "times." )
"""

#print("Symmetrized little omega")

non_zero_lo=np.zeros((omega_size+1),dtype=int)
for i in range (1,omega_size+1):
    non_zero_lo[i]=0
    for j in range (little_omega_size+1):
        for k in range (j,little_omega_size+1):
            if (z[i,j,k])!= 0. :
                non_zero_lo[i]+=1

max_nonzero_size_lo = np.amax(non_zero_lo)

""" inserted here
for i in range (1,omega_size+1):
    for j in range (1,omega_size+1):
        non_zero[i,j]=0
        for k in range (max_size+1):
            if (y[i][j][k])!= 0. :
                non_zero[i,j]+=1

max_nonzero_size = np.amax(non_zero)
"""

print()
print("Maximum number of loop pairs into which any given loop breaks :")
print(max_nonzero_size_lo)



nonzero_loop1_index=np.zeros((omega_size+1, max_nonzero_size_lo),dtype=int)
nonzero_loop2_index=np.zeros((omega_size+1, max_nonzero_size_lo),dtype=int)
nonzero_z=np.zeros((omega_size+1, max_nonzero_size_lo),dtype=int)
for i in range (1,omega_size+1):
    indx=0
    for j in range (little_omega_size+1):
        for k in range (j,little_omega_size+1):
            if (z[i,j,k])!= 0. :
                nonzero_loop1_index[i,indx]=j
                nonzero_loop2_index[i,indx]=k
                nonzero_z[i,indx]=z[i,j,k]
                indx+=1

print ()
print("Do you want to print how loops are split  ?")
yes_or_no = "w"
while not (yes_or_no=="y" or yes_or_no=="n"):
    yes_or_no=input ("Enter 'y' or 'n' :    ")


if yes_or_no=="y":
    print()
    for i in range (1,omega_size+1):
        for indx in range(non_zero_lo[i]):
            print ("Loop", i,"=",loop_list[i],"breaks into", "Loop", nonzero_loop1_index[i,indx],"=", loop_list[nonzero_loop1_index[i,indx]], "and Loop", nonzero_loop2_index[i,indx], "=", loop_list[nonzero_loop2_index[i,indx]],nonzero_z[i,indx], "times." )
            #print ("No. splittings ", non_zero_lo[i], "Loop", i,"=",loop_list[i],"breaks into", "Loop", nonzero_loop1_index[i,indx],"=", loop_list[nonzero_loop1_index[i,indx]], "and Loop", nonzero_loop2_index[i,indx], "=", loop_list[nonzero_loop2_index[i,indx]],nonzero_z[i,indx], "times." )

print("\n")
import os
import pathlib
from pathlib import Path


PATH = pathlib.Path(__file__).parent.absolute() #get the parent directory of the script
current_folder = str(PATH)
print("PATH = ", current_folder)
data_folder_name = "data1"
data_folder = os.path.join(current_folder, data_folder_name)
print("Data folder is :", data_folder)
data_file_name = "lmax_"+str(omega_length)+"_collective_info_v2.npy"
data_file = os.path.join(data_folder, data_file_name)
os.makedirs(data_folder,  exist_ok=True)
with open(data_file, 'wb') as f:
    np.save(f,omega_length)
    np.save(f,max_length)
    np.save(f,numb_tuples_list)
    np.save(f,loop_list)
    np.save(f,non_zero)
    np.save(f,nonzero_index)
    np.save(f,nonzero_y)
    np.save(f,non_zero_lo)
    np.save(f,nonzero_loop1_index)
    np.save(f,nonzero_loop2_index)
    np.save (f, nonzero_z)
    np.save(f, adjoint_loops)

with open(data_file, 'rb') as f:
    loaded_omega_length=np.load(f)
    loaded_max_length=np.load(f)
    loaded_numb_tuples_list=np.load(f,allow_pickle=True)
    loaded_loop_list=np.load(f,allow_pickle=True)
    #loaded_y=np.load(f,allow_pickle=True)
    loaded_non_zero=np.load(f)
    loaded_nonzero_index=np.load(f)
    loaded_nonzero_y=np.load(f)
    loaded_non_zero_lo=np.load(f)
    loaded_nonzero_loop1_index=np.load(f)
    loaded_nonzero_loop2_index=np.load(f)
    loaded_nonzero_z=np.load(f)
    #loaded_z = np.load(f,allow_pickle=True)
    loaded_adjoint_loops = np.load(f, allow_pickle=True)


print()
print ("Loops, splitting and joining information stored in file "+"lmax_"+str(omega_length)+"_collective_info_v2.npy")
print()
print ("Info extracted from file")
print ("lmax = ", loaded_omega_length)
print ("max length =", loaded_max_length)
print ("loaded_numb_tuples_list = \n", loaded_numb_tuples_list)
print ("loaded_adjoint_loops: ", loaded_adjoint_loops)
print()
max_size = loaded_numb_tuples_list [loaded_max_length][1]
print ("Total number of loops up to length ", loaded_max_length, ": ", max_size)
print ("List of loops :")
i=0
for x in loaded_loop_list:
    print ("Loop", i,"=", x)
    i+=1
for i in range (loaded_max_length+1):
    num=loaded_numb_tuples_list[i][1]-loaded_numb_tuples_list[i][0]+1
    print ("Number of loops of length " ,i, " : ", num,)
#print (loaded_loop_list)
print()
omega_size = loaded_numb_tuples_list [omega_length][1]

"""
for i in range (1,omega_size+1):
    for j in range (1,omega_size+1):
        for k in range (max_size+1):
            if (loaded_y[i][j][k])!= 0. :
                print ("Loop", i,"=",loaded_loop_list[i],"joins with", "Loop", j,"=", loaded_loop_list[j], "into Loop", k,"=", loaded_loop_list[k], (loaded_y[i][j][k]), "times." )
"""
print ()
print("Do you want to print splitting and joining information saved to the file?")
yes_or_no = "w"
while not (yes_or_no=="y" or yes_or_no=="n"):
    yes_or_no=input ("Enter 'y' or 'n' :    ")

if yes_or_no=='y':

    for i in range (1,omega_size+1):
        for j in range (1,omega_size+1):
            #print ("range", non_zero[i,j])
            for indx in range(loaded_non_zero[i,j]):
                print ("Loop", i,"=",loaded_loop_list[i],"joins with", "Loop", j,"=", loaded_loop_list[j], "into Loop", loaded_nonzero_index[i,j,indx],"=", loaded_loop_list[loaded_nonzero_index[i,j,indx]], loaded_nonzero_y[i,j,indx], "times." )

    print ()

    for i in range (1,omega_size+1):
        for indx in range(loaded_non_zero_lo[i]):
            print ("Loop", i,"=",loaded_loop_list[i],"breaks into", "Loop", loaded_nonzero_loop1_index[i,indx],"=", loaded_loop_list[nonzero_loop1_index[i,indx]], "and Loop", loaded_nonzero_loop2_index[i,indx], "=", loaded_loop_list[nonzero_loop2_index[i,indx]],loaded_nonzero_z[i,indx], "times." )

"""
little_omega_size=loaded_numb_tuples_list [loaded_omega_length-2][1]
for i in range (1,omega_size+1):
    for j in range (little_omega_size+1):
        for k in range (little_omega_size+1):
            if (loaded_z[i][j][k])!= 0. :
                print ("Loop", i,"=",loaded_loop_list[i],"breaks into", "Loop", j,"=", loaded_loop_list[j], "and Loop", k,"=", loaded_loop_list[k], (loaded_z[i][j][k]), "times." )
"""

#for y in loaded_master_field :
    #print (y)

