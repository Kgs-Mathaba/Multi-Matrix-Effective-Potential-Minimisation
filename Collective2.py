import numpy as np

def foo(arg):
    print('arg = {}'.format(arg))

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

def cyclic_generator(lp):

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

def load_npy_data(data_file):
    with open(data_file, 'rb') as f:
        omega_length=np.load(f)
        max_length=np.load(f)
        numb_tuples_list=np.load(f,allow_pickle=True)
        loop_list=np.load(f,allow_pickle=True)
        non_zero=np.load(f,allow_pickle=True)
        nonzero_index=np.load(f,allow_pickle=True)
        nonzero_y=np.load(f,allow_pickle=True)
    #old_y=np.load(f,allow_pickle=True)
        non_zero_lo=np.load(f,allow_pickle=True)
        nonzero_loop1_index=np.load(f,allow_pickle=True)
        nonzero_loop2_index=np.load(f,allow_pickle=True)
        nonzero_z=np.load(f,allow_pickle=True)
    #old_z = np.load(f,allow_pickle=True)
        adjoint_loops = np.load(f, allow_pickle=True)

        data = (omega_length,
                max_length,
                numb_tuples_list,
                loop_list,
                non_zero,
                nonzero_index,
                nonzero_y,
                non_zero_lo,
                nonzero_loop1_index,
                nonzero_loop2_index,
                nonzero_z,
                adjoint_loops)

    return data


if __name__== "__main__":

    loop, omega, little_omega, LnJ = initialize_loops(4, 4)
    print("loop = ", loop)

    elements = ('a', 'b', 'c', 'd', 'e')
    type(elements)
    
    for index, element in enumerate(elements):
        print(element, index)