
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


using Eigen::VectorXd; // Vector of doubles with dynamic size
using Eigen::MatrixXd; // Matrix of doubles with dynamic size
using namespace LBFGSpp; 
using namespace Eigen;
using namespace std;


int main()
{

	// set  some constants
	const int n_thooft = 10;
    const int n_thooft_squared = pow(n_thooft,2);
    const int omega_length = 4;
    const int max_length = 2*omega_length -2; 
    int start = 0;
    int end = 0;



	// Initialiizing the vector of vectors (dynamical array) loop_list
    std::vector<std::vector<int> > loop_list;


    // loop to fill loop_list
    for(int i = 1; i < max_length+1; i++){
    	int row_index = i; // row_index elements
    	int col_index = i+1; // col_indexumn elements

    	cout << "i = " << i << endl ;
    	//initializing 2d array loops_fixed_length
    	
    	int loops_fixed_length[row_index][col_index];
    	
    	
    	// loop to fill loops_fixed_length
    	for(int c=0; c < col_index; c++){
    		for (int r = 0; r < row_index; ++r)
    		{
    			if(r< i-c){loops_fixed_length[r][c]=1;}
    			else{loops_fixed_length[r][c]=2;}
    		}
    	}


    	cout << "loops_fixed_length = \n";
    	// print loops_fixed_length
    	for(int c=0; c < col_index; c++){
    		for(int r=0; r < row_index; r++){
    			
    			cout << loops_fixed_length[r][c] << " " ;
    		}
    		// Newline for new row_index
   			cout <<  endl;
    	}
    	cout << endl;


    	// create loops from loops_fixed_length
    	for(int c=0; c < col_index; c++)
    	{
    		//for each row_index of loops_fixed_length create loop (vector)
    		std::vector<int> loop;
    		// fill loop with row_index elements of loops_fixed_length
    		for (int r = 0; r < row_index; ++r)
    		{
    			loop.push_back( loops_fixed_length[r][c] );
    		}
    		// add loop to loop_list
    		loop_list.push_back(loop);
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