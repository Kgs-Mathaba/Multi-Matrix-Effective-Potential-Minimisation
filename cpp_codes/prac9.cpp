#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>
#include <chrono> 
#include <fstream>


using Eigen::VectorXd; // Vector of doubles with dynamic size
using Eigen::MatrixXd; // Matrix of doubles with dynamic size
using namespace LBFGSpp; 
using namespace Eigen;
using namespace std;
using namespace std::chrono; 





double Effect_pot(const VectorXd& Eigen, VectorXd& grad )
{
    
	const int NtHooft = Eigen.size();
	const int OmegaSize = 10;					//lmax
	int OmegaLength = 2*OmegaSize-2;	//M
	
	
	// Obtain loops directly from eigenvalues
	ArrayXd Eigen_array = Eigen.array();
	VectorXf loop(OmegaLength +1);
	
	for(int i = 0; i < OmegaLength + 1 ; i++)
	{
        loop[i] = Eigen_array.pow(i).sum()/pow(NtHooft,((i+2)/2) );
        cout << "loop[" << i <<"] = " <<loop[i] << endl<<endl;
	};
        
	// Construct little omega
	VectorXf omeg =VectorXf::Zero(OmegaSize);
	for(int i=1; i < OmegaSize; i++)
	{
		for(int j=0; j < i; j++)
		{
			omeg[i] += (i+1)*loop[j]*loop[i-1-j];	
		};
	};
	
	cout << "Here is little omega: "<< endl << endl << omeg << endl<<endl;
	
		// Construct Omega
	MatrixXf Omeg(OmegaSize,OmegaSize);
	for (int i = 0; i < OmegaSize; ++i)
	{
		for (int j = 0; j < OmegaSize; ++j)
		{
			Omeg(i,j) = (i+1)*(j+1)*loop[i+j];
		};
	};

	cout<< "Here is Omega: "<< endl << endl << Omeg << endl<<endl;
	
	
	
	/*

	Cholesky decomposition of a Hermitian positive-definite matrix A, is a decomposition of the form
	A = LL*
	where L is a lower triangular matrix with real and positive diagonal entries, and L* denotes the 	conjugate transpose of L 
	*/


	VectorXf LnJ = Omeg.llt().solve(omeg);  // compute the Cholesky decomposition of Omega and solve 	Omega*LnJ = omega 
	cout << "Here is LnJ: " << endl << endl << LnJ << endl;

	
	// return the Large N Energy
	double LargeNEnergy = omeg.dot(LnJ)/8 + loop[2]/2;
	cout << "Here is the Large N Energy: " << endl << endl << LargeNEnergy << endl<<endl;



	// Derivatives directly with respect to eigenvalues
	grad = VectorXd::Zero(NtHooft);	
	for(int n=0; n < NtHooft; n++)
	{
		for(int i=2; i < OmegaSize; i++)
		{
			for(int l=1; l < i; l++)
			{
				grad[n] += 4*(i+1)*l*pow(Eigen_array[n],(l-1))*loop[i-1-l] *LnJ[i] /(8*pow(NtHooft,((l+2)/2)));
			}
		}
	};
	 
	 for(int n=0; n < NtHooft; n++)
	{
		for(int i=1; i < OmegaSize; i++)
		{
			for(int j=0; j < OmegaSize; j++)
			{
				grad[n] += -LnJ[i]*(i+1)*(j+1)*(i+j)*pow(Eigen_array[n],(i+j-1))*LnJ[j]/(8*pow(NtHooft,((i+j+2)/2) )) ;
			}
		}
	};
	 
	 
	 for(int n=0; n < NtHooft; n++)
	 {
	 	for(int j=1; j < OmegaSize; j++)
	 	{
	 		grad[n] += -LnJ[0]*(j+1)*(j)*pow(Eigen_array[n],(j-1))*LnJ[j]/(8*pow(NtHooft,((j+2)/2)));
	 	}
	 };
	 
	 for(int n=0; n < NtHooft; n++)
	 {
	 	grad[n] += Eigen_array[n]/pow(NtHooft,2);
	 };


	cout << "Here is effect_pot grad: " << endl << endl << grad << endl<<endl;
	


	 
 
    return LargeNEnergy;
    
}

int main()
{
    int n=60;
	//cout<<"Enter the value of NtHooft: ";
	//cin>>n;
	
	
	
	
    // Initial guess
    VectorXd Eigen = VectorXd::LinSpaced(Sequential, n,-0.5,0.5)*sqrt(n);
    cout << "Here is EigenInit = " << endl << Eigen << endl<<endl;
    
    //VectorXd grad = VectorXd::Zero(n);
     
     
    //double z;
    //z =  Effect_pot(Eigen , grad);
	//cout << "Here is the foo at x = " << endl << z << endl;
	
	LBFGSParam<double> param;
    param.epsilon = 1e-16;
    param.max_iterations = 0;
    LBFGSSolver<double> solver(param);
     
    


	// Get starting timepoint 
    auto start = std::chrono::steady_clock::now();
  
    
    
	double LargeNEnergy;
	int niter = solver.minimize(Effect_pot, Eigen , LargeNEnergy);
	cout << "Optimization terminated successfully." << endl;
	cout << "Current function value: " << LargeNEnergy << endl;
	cout << niter << " iterations" << std::endl;
	
	
	
    // Get ending timepoint 
    auto end = std::chrono::steady_clock::now();
	
	
	// Get duration. Substart timepoints to  
    // get durarion. To cast it to proper unit 
    // use duration cast method 
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	
	
	
    return 0;
}
