# Single Hermitian Matrix
# Hamiltonian Loop Space

# Load library
import numpy as np
from scipy.optimize import minimize

"""
    EffectPot is a function that returns the large N Energy.
    This is the value of minimum of the collective effective potential.
    The function uses a minimization method requiring the gradient as well.
    This is given in the function Grad
        Inputs:
            Eigenvalues (Array/vector of dimension NtHooft);
            OmegaSize (Omega is a OmegaSize x OmegaSize Matrix)
                The largest loop generated has lentgth 2 * OmegaSize - 2;
            NtHooft ( N of large N)
            gcoupling (coupling of quartic interaction gcoupling * \phi^4).
    Loop , omeg (little omega) and Omega are global variables.
    Note: in Python, the first index is 0, not 1.
"""
def EffectPot(Eigen, OmegaSize, NtHooft, gcoupling):

    global Loop
    global Omega
    global omeg

    # Obtain Loops from eigenvalues.
    for i in range(2*OmegaSize-1) :
        Loop[i]=np.sum (np.power(Eigen,i)) / (NtHooft**((i+2)/2))

    # Generate Omega matrix from Loops
    Omega = [[(i+1)*(j+1)*Loop[i+j] for i in range(OmegaSize)]for j in range(OmegaSize)]

    # Generate "little omega" (omeg) from Loops
    omeg = np.zeros (OmegaSize)
    for i in range(1, OmegaSize):
        omeg [i] = np.sum ([(i+1)*Loop[j]*Loop[i-1-j] for j in range(i)])

    # Solve system of linear equations for LnJ, instead of inverting Omega
    LnJ = np.linalg.solve (Omega, omeg)

    # Obtain value of collective effective potential
    LargeNEnergy = ( np.dot (omeg , LnJ)/8.0 + Loop[2]/2. + gcoupling*Loop[4] )

    return LargeNEnergy

def Grad (Eigen, OmegaSize, NtHooft, gcoupling):

    global GradLoop
    global GradOmega
    global Gradomeg

    # Obtain original Loops from eigenvalues
    for i in range(2*OmegaSize-1) :
        GradLoop[i]=np.sum (np.power(Eigen,i)) / (NtHooft**((i+2)/2))

    # Generate Omega matrix from Loops
    GradOmega = [[(i+1)*(j+1)*GradLoop[i+j]for i in range(OmegaSize)]for j in range(OmegaSize)]

    # Generate "little omega" (omeg) from Loops
    Gradomeg = np.zeros (OmegaSize)
    for i in range(1, OmegaSize):
        Gradomeg [i] = np.sum ([(i+1)*GradLoop[j]*GradLoop[i-1-j] for j in range(i)])

    # Solve system of linear equations for LnJ, instead of inverting Omega
    GradLnJ = np.linalg.solve (GradOmega, Gradomeg)

    # Derivatives directly with respect to eigenvalues
    Deriv = np.zeros(NtHooft)
    for n in range(NtHooft):
        for i in range(2,OmegaSize):
            for l in range(1,i):
                Deriv[n]+=  (  4*(i+1)*l*np.power(Eigen[n],(l-1))*GradLoop[i-1-l]*GradLnJ[i]  ) / (8*NtHooft**((l+2)/2))
        Deriv[n]+= - np.sum ([[GradLnJ[i]*(i+1)*(j+1)*(i+j)*np.power(Eigen[n],(i+j-1))*GradLnJ[j]/(NtHooft**((i+j+2)/2))for i in range(1,OmegaSize)]for j in range(OmegaSize)])/8
        Deriv[n]+= - np.sum ([GradLnJ[0]*(j+1)*(j)*np.power(Eigen[n],(j-1))*GradLnJ[j]/(NtHooft**((j+2)/2))for j in range(1,OmegaSize)])/8
        # Potential
        Deriv[n]+= (Eigen[n] / (NtHooft**2) + 4*gcoupling*np.power(Eigen[n],3)/(NtHooft**3))

    return Deriv



# Main code

# Input values for g, size of Omega and number of eigenvalues (N)
# If no convergence is achieved, N is increased by 10

gstring = input ("Enter a value for the coupling:   ")
g = float(gstring)

OmegaString = input ("Enter the size of Omega:   ")
OmegaSizeInit = int(OmegaString)

NString = input ("Enter minimum number of eigenvalues:    ")
N = int(NString)

print ('N will be increased in steps of 10, if necessary, until convergence is achieved')

Converge = False

while Converge == False:
    # Initialize N eigenvalues. Distributed uniformly between -0.5 and 0.5
    EigenInit = np.linspace(-0.5,0.5,N)*np.sqrt(N)
    #EigenInit =(np.random.rand(N)-np.ones(N)*0.5)*np.sqrt(N)

    # Initialize global variables: Loops, Omega and omega to zero for all entries
    Loop = np.zeros (2*OmegaSizeInit-1)
    Omega = np.zeros ((OmegaSizeInit, OmegaSizeInit))
    omeg = np.zeros (OmegaSizeInit)

    # Initialize global variables for gradient function
    GradLoop = np.zeros (2*OmegaSizeInit-1)
    GradOmega = np.zeros ((OmegaSizeInit, OmegaSizeInit))
    Gradomeg = np.zeros (OmegaSizeInit)

    # Minimization method is BFGS, with gradient
    Energy = minimize (EffectPot, EigenInit, args=(OmegaSizeInit, N, g), method='BFGS', jac=Grad, options={'disp' : True, 'maxiter': 5000})

    Converge = Energy.success
    if Converge == True:
        print ("Minimization has converged")
        print ("Energy = ", round (Energy.fun,6))
        for i in range(2*OmegaSizeInit-1):
            print ("Loop ,",i,",",round(np.sum (np.power(Energy.x,i)) / (N**((i+2)/2)),6))
        break
    else:
        N+=10
        if N>1000: break
        print ("Trying N = ", N)
