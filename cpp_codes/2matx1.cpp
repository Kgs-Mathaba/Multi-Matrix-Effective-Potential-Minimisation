#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>
#include <algorithm>
#include <random>
#include<chrono>
#include <vector>


using Eigen::VectorXd; // Vector of doubles with dynamic size
using Eigen::MatrixXd; // Matrix of doubles with dynamic size
using namespace LBFGSpp; 
using namespace Eigen;
using namespace std;



// Utility to print vectors
void print(std::vector <int> const &a) {
   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';
}



// Function to check if an array is
// subarray of another array
bool is_cyclic_perm(std::vector<int>  A, std::vector<int>  B)
{
    // Two pointers to traverse the arrays
    int i = 0, j = 0;

    auto m = A.size();
    auto n = B.size();

    // Traverse both arrays simultaneously
    while (i < m&& j < n) {

        // If element matches
        // increment both pointers
        if (A[i] == B[j]) {

            i++;
            j++;

            // If array B is completely
            // traversed
            if (j == n)
                return true;
        }
        // If not,
        // increment i and reset j
        else {
            i = i - j + 1;
            j = 0;
        }
    }

    return false;
}




int main(){


	// set  some constants
	const int n_thooft = 10;
    const int n_thooft_squared = pow(n_thooft,2);
    const int omega_length = 3;
    const int max_length = 2*omega_length -2; 
    int start = 0;
    int end = 0;

	/* Create random engine with the help of seed */
    unsigned seed = chrono::steady_clock::now().time_since_epoch().count(); 
    default_random_engine e (seed); 
  
    /* declaring normal distribution object 'distN' and initializing its mean and standard deviation fields. */
  	/* Here, we have used mean=0, and standard deviation=1. */
    normal_distribution<double> distN(0,1); 
    
    


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



	// Initializing the vector of vectors (dynamical array) loop_list
    std::vector<std::vector<int> > loop_list;


    // loop to fill loop_list
    for(int i = 1; i < max_length+1; i++){
    	int row_index = i; // row elements
    	int col_index = pow(2,i); // col elements

    	cout << "i = " << i << endl ;
    	//initializing 2d array binary_array
    	
    	int binary_array[row_index][col_index];
    	
    	
    	// loop to fill binary_array
    	for(int c=0; c < col_index; c++){
            int n=c;
    		for (int r = 0; r < row_index; ++r)
    		{
    			binary_array[row_index-(r+1)][c]=n%2 + 1;
                n=n/2;
    		}
    	}

        // Display binary array
    	cout << "binary_array = \n";
    	// print binary_array
    	for(int c=0; c < col_index; c++){
    		for(int r=0; r < row_index; r++){
    			
    			cout << binary_array[r][c] << " " ;
    		}
    		// Newline for new row
   			cout <<  endl;
    	}
    	cout << endl;

        


    	// create loops from binary_array
    	for(int c=0; c < col_index; c++)
    	{
    		//for each row_index of binary_array create loop (vector)
    		std::vector<int> loop;

            


    		// fill loop with row_index elements of binary_array
    		for (int r = 0; r < row_index; ++r)
    		  {loop.push_back( binary_array[r][c] );}

            // print loop
            cout << "loop = ";
            print(loop);
            cout << endl;

            
            // from third loop run comparison
            std::vector<int> loop1_concat = loop;
            loop1_concat.insert(loop1_concat.end(), loop.begin(), loop.end());
            
             
            /*
            Check if loop is a cyclic perm of previous loops by concatenating it by itself.
            Then check if any previous loops are contained.
            */
            
            bool result = false;
            if(c>1){
                    //start loop over previous arrays
                    for(int d=1; d<c; d++){

                        //create and fill loop2
                        std::vector<int> loop2;
                        for (int r = 0; r < row_index; ++r)
                        {loop2.push_back( binary_array[r][d] );}
                        cout << "loop2 = ";
                        print(loop2);
                        cout << endl;

                        
                        result = is_cyclic_perm(loop1_concat, loop2);
                        if(result){

                            break;
                        }
                    }
                }

    		// add loop to loop_list
            if(!result)
    		{loop_list.push_back(loop);}
    	   }
     }

    
    // Displaying the 2D vector loop_list
    cout << "loop_list = \n";
    for (int i = 0; i < loop_list.size(); i++) {

    	cout << "loop[" << i+1 <<"] = [ ";
        for (int j = 0; j < loop_list[i].size(); j++){
            cout  <<loop_list[i][j] << " ";
        }
        cout << "]"<< endl;
    }

    int max_size = loop_list.size()+1;
    cout << "total number of loops = " << max_size << endl;






	return 0;
}
