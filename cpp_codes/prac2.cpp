
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace LBFGSpp;
using namespace Eigen;
using namespace std;

int main(){



VectorXd x = VectorXd::Random(10);
cout << "here is the random vector x: " << endl << x;

const int n = x.size();
    VectorXd loopy(n);
	
    for(int i = 0; i < n; i++)
        loopy[i] = i;
	cout << "Here is the loopy array: "<<endl<<loopy;

//MatrixXf mat(20,1);
//mat.setRandom();
//cout << "here is the random matrix mat: " << endl << mat;
//MatrixXf mat2 = mat.array().abs(); // also mat.cwiseAbs()
//MatrixXf mat3 = mat.array().pow(0);



//VectorXf w(n);
//w.setRandom();
VectorXf w = VectorXf::Random(10);
VectorXf w1 = VectorXf::Random(10);
cout<<"here is the random vector w: "<< endl << w1;
ArrayXf w2 = w1.array().pow(2);
cout<<"here is the vector w raised to power of 2: "<< endl << w2<<endl;
float w3 = w2.sum();
cout<<"here is the sum of the elements of w raise to the power of two : "<< endl << w3 << endl;
VectorXf loop(2*n-1);

for(int i = 0; i < 2*n-1; i++)
        loop[i] = w.array().pow(i).sum()/pow(n,((i+2)/2) );
	cout << "Here is the loop array: "<< endl << loop << endl;




VectorXf omeg(n);
for(int i=0; i < n; i++)
{
	for(int j=0; j < i; j++)
	{
		omeg[i] = (i+1)*loop[i]*loop[i-1-j];	
	}
}


cout << "Here is little omega: "<< endl<< omeg << endl;

cout << "Here is n: "<< endl<< n << endl;


MatrixXf Omeg(n,n);
	for (int i = 0; i < 10; ++i)
	{
		for (int j = 0; j < 10; ++j)
		{
			Omeg(i,j) = (i+1)*(j+1)*loop[i+j];
		}
	}
cout<< "Here is Omega: "<< endl << Omeg << endl;




MatrixXd A = MatrixXd::Random(3,3);
cout << "The matrix A is" << endl << A << endl << endl;
 
LLT<MatrixXd> lltOfA(A); // compute the Cholesky decomposition of A
MatrixXd L = lltOfA.matrixL(); // retrieve factor L  in the decomposition
// The previous two lines can also be written as "L = A.llt().matrixL()"
 
cout << "The Cholesky factor L is" << endl << L << endl;
cout << "To check this, let us compute L * L.transpose()" << endl << endl;
cout << L * L.transpose() << endl;
cout << "This should equal the matrix A" << endl;



/*

Cholesky decomposition of a Hermitian positive-definite matrix A, is a decomposition of the form
A = LL*
where L is a lower triangular matrix with real and positive diagonal entries, and L* denotes the conjugate transpose of L 
*/


VectorXf LnJ = Omeg.llt().solve(omeg);  // compute the Cholesky decomposition of Omega and solve Omega*LnJ = omega 
cout << "Here is LnJ: " << endl << LnJ << endl;

cout << "Here is the size of LnJ: " << LnJ.size() << endl;




float LargeNEnergy = omeg.dot(LnJ);



cout << "Here is the Large N Energy: " << endl << LargeNEnergy << endl;




// Derivatives directly with respect to eigenvalues
VectorXf grad = VectorXf::Zero(n);
for(int i=0; i < n; i++)
{
	for(int j=2; j < 2*n-1; j++)
	{
		for(int k=1; k < j; k++)
		{
			grad[i] += 4*(j+1)*k*pow(w.array()[i],k);		
		}	
	}
}


cout << "Here is grad: " << endl << grad << endl;




return 0;
}
       
