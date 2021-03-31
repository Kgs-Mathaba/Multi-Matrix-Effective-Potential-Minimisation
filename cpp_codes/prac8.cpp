#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>

using Eigen::VectorXd; // Vector of doubles with dynamic size
using Eigen::MatrixXd; // Matrix of doubles with dynamic size
using namespace LBFGSpp; 
using namespace Eigen;
using namespace std;




double Effect_pot(const VectorXd& Eigen, VectorXd& grad)
{
    
	const int NtHooft = Eigen.size();
	const int OmegaSize = 8;					//lmax
	int OmegaLength = 2*OmegaSize-2;	//M
	
	
	// Obtain loops directly from eigenvalues
	ArrayXd Eigen_array = Eigen.array();
	VectorXf loop(OmegaLength +1);
	
	for(int i = 0; i < OmegaLength + 1 ; i++)
	{
        loop[i] = Eigen_array.pow(i).sum()/pow(NtHooft,((i+2)/2) );
        cout << "loop[" << i <<"] = " <<loop[i] << endl;
	};
        
	// Construct little omega
	VectorXf omeg(OmegaSize);
	for(int i=0; i < OmegaSize; i++)
	{
	for(int j=0; j < i; j++)
	{
		omeg[i] = (i+1)*loop[i]*loop[i-1-j];	
	};
	};
	
	cout << "Here is little omega: "<< endl<< omeg << endl;
	
		// Construct Omega
	MatrixXf Omeg(OmegaSize,OmegaSize);
	for (int i = 0; i < OmegaSize; ++i)
	{
		for (int j = 0; j < OmegaSize; ++j)
		{
			Omeg(i,j) = (i+1)*(j+1)*loop[i+j];
		};
	};

	cout<< "Here is Omega: "<< endl << Omeg << endl;
	
	
	
	/*

	Cholesky decomposition of a Hermitian positive-definite matrix A, is a decomposition of the form
	A = LL*
	where L is a lower triangular matrix with real and positive diagonal entries, and L* denotes the 	conjugate transpose of L 
	*/


	VectorXf LnJ = Omeg.llt().solve(omeg);  // compute the Cholesky decomposition of Omega and solve 	Omega*LnJ = omega 
	cout << "Here is LnJ: " << endl << LnJ << endl;

	
	// return the Large N Energy
	double LargeNEnergy = omeg.dot(LnJ);
	cout << "Here is the Large N Energy: " << endl << LargeNEnergy << endl;



	// Derivatives directly with respect to eigenvalues
	VectorXf effect_pot_grad = VectorXf::Zero(NtHooft);	
	for(int n=0; n < NtHooft; n++)
	{
		for(int i=2; i < OmegaSize; i++)
		{
			for(int l=1; l < i; l++)
			{
				effect_pot_grad[n] += 4*(i+1)*l*pow(Eigen_array[n],(l-1))*loop[i-1-l] *LnJ[i] /(8*pow(NtHooft,((l+2)/2)));
			}
		}
	};
	 
	 for(int n=0; n < NtHooft; n++)
	{
		for(int i=1; i < OmegaSize; i++)
		{
			for(int j=0; j < OmegaSize; j++)
			{
				effect_pot_grad[n] += -LnJ[i]*(i+1)*(j+1)*(i+j)*pow(Eigen_array[n],(i+j-1))*LnJ[j]/(8*pow(NtHooft,((i+j+2)/2) )) ;
			}
		}
	};
	 
	 
	 for(int n=0; n < NtHooft; n++)
	 {
	 	for(int j=1; j < OmegaSize; j++)
	 	{
	 		effect_pot_grad[n] += -LnJ[0]*(j+1)*(j)*pow(Eigen_array[n],(j-1))*LnJ[j]/(8*pow(NtHooft,((j+2)/2)));
	 	}
	 };


	cout << "Here is effect_pot grad: " << endl << effect_pot_grad << endl;
	


	 
 
    return LargeNEnergy;
    
}

int main()
{
    int n;
	cout<<"Enter the value of NtHooft: ";
	cin>>n;

    VectorXd Eigen = VectorXd::Random(n);
    cout << "Here is EigenInit = " << endl << Eigen << endl;
    VectorXd grad = VectorXd::Zero(n);
	cout << "Here is grad = " << endl << grad << endl;
    double z;
    z =  Effect_pot(Eigen , grad);
	cout << "Here is the foo at x = " << endl << z << endl;

    return 0;
}
