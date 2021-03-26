#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>
#include <algorithm>
#include <random>
#include<chrono>


using Eigen::VectorXd; // Vector of doubles with dynamic size
using Eigen::MatrixXd; // Matrix of doubles with dynamic size
using namespace LBFGSpp; 
using namespace Eigen;
using namespace std;

int main(){


	/* Create random engine with the help of seed */
    unsigned seed = chrono::steady_clock::now().time_since_epoch().count(); 
    default_random_engine e (seed); 
  
    /* declaring normal distribution object 'distN' and initializing its mean and standard deviation fields. */
  	/* Here, we have used mean=0, and standard deviation=1. */
    normal_distribution<double> distN(0,1); 
    
    const int  n_thooft = 10;
    const int n_thooft_squared = pow(n_thooft,2);
    //cout<<"\nEnter size of n_thooft: ";
    //cin>>n_thooft;
    cout<<"\nNormal distribution for "<<n_thooft<<" samples (mean=0, standard deviation=1) =>\n";


    // initialize vector of doubles and fill with randoms generated from normal distribution.
    VectorXd x(n_thooft*(n_thooft+1));
    for (int i=0; i<n_thooft*(n_thooft+1); i++)
    {
      x[i] = distN(e);

    }   
    


	

	VectorXd x_init = (x- (VectorXd::Ones(n_thooft*(n_thooft+1)))*0.5 )*sqrt(n_thooft); 
	cout << "x_init = \n" << x_init << endl;




	// slice last n_thooft^2 elements and build temp matrix
	Eigen::Matrix<double, 1, n_thooft_squared > x2 = x_init.tail(pow(n_thooft,2));
	Map<Eigen::Matrix<double, n_thooft, n_thooft >> temp_matrix_2(x2.data());
	cout << "temp_matrix_2 = \n" << temp_matrix_2 << endl;
	MatrixXd temp_matrix_3 = MatrixXd(temp_matrix_2.triangularView<Lower>());
	cout << "temp_matrix_3 = \n" << temp_matrix_3 << endl;
	MatrixXd temp_matrix_4 = temp_matrix_3 +temp_matrix_3.transpose() ;
	cout << "temp_matrix_4 = \n" << temp_matrix_4 << endl;
	
	VectorXd v1 = temp_matrix_3.diagonal();
	cout <<"v1 = \n" <<v1 << endl;
	MatrixXd mat1 = v1.asDiagonal();
	cout << "mat1 \n" << mat1 << endl;


	MatrixXd real_matrix_2 = temp_matrix_3 +temp_matrix_3.transpose() -  mat1;
	cout << "real_matrix_2 = \n" << real_matrix_2 << endl;

	MatrixXd temp_matrix_5 = MatrixXd(temp_matrix_2.triangularView<StrictlyUpper>());
	cout << "temp_matrix_5 = \n" << temp_matrix_5 << endl;
	MatrixXd imag_matrix_2 = temp_matrix_5 - temp_matrix_5.transpose();
	cout << "imag_matrix_2 = \n" << imag_matrix_2 << endl;


	//build a complex matrix
 	Eigen::MatrixXcd m(2,2);
    m << 2.0 + 2.0if, 2.0f + 1.0if, 3.0f - 1.0if, 4.0f - 2.0if;
    std::cout << m << std::endl;
	
    // Second matrix
	MatrixXcd matrices_array_1(n_thooft,n_thooft);
    for(int i=0; i<n_thooft; i++){
    	for(int j=0; j<n_thooft; j++){
    		matrices_array_1(i,j) = real_matrix_2(i,j)*1.0f + imag_matrix_2(i,j)*1.0if;
    	}
    }
    cout << "matrices_array_1 = \n" << matrices_array_1 << endl;

    Eigen::Matrix<double, 1, n_thooft > x4 = x_init.head(n_thooft);
    cout << "x4 = \n" << x4.transpose() << endl;


    // First matrix is a diagonal matrix
    MatrixXcd matrices_array_0 = x4.asDiagonal();
    cout << "matrices_array_0 = \n" << matrices_array_0 << endl;






	return 0;
}