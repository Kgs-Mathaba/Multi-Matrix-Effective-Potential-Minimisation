
// C++ program to demonstrate insertion
// into a vector of vectors
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <LBFGS.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <vector>
#include <bits/stdc++.h>


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






int main()
{

	
    // set  some constants
	const int n_thooft = 10;
    const int n_thooft_squared = pow(n_thooft,2);
    const int omega_length = 4;
    const int max_length = 2*omega_length -2; 
    int start = 0;
    int end = 0;



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

    // Build loops
    

    return 0;
}