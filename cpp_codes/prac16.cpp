
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
    const int omega_length = 3;
    const int max_length = 2*omega_length -2; 
    int start = 0;
    int end = 0;



	// Initialiizing the vector of vectors (dynamical array) loop_list
    std::vector<std::vector<int> > loop_list;

    // loop to fill loop_list
    for(int i = 1; i < max_length+1; i++){
    	int Row = i; // Row elements
    	int Col = i+1; // column elements

    	cout << "i = " << i << endl ;
    	//initializing 2d array loops_fixed_length
    	
    	int loops_fixed_length[Row][Col];
    	
    	
    	// loop to fill loops_fixed_length
    	for(int c=0; c < Col; c++){
    		for (int r = 0; r < Row; ++r)
    		{
    			if(r< i-c){loops_fixed_length[r][c]=1;}
    			else{loops_fixed_length[r][c]=2;}
    		}
    	}


    	cout << "loops_fixed_length = \n";
    	// print loops_fixed_length
    	for(int c=0; c < Col; c++){
    		for(int r=0; r < Row; r++){
    			
    			cout << loops_fixed_length[r][c] << " " ;
    		}
    		// Newline for new row
   			cout <<  endl;
    	}
    	cout << endl;
    	// create loops from loops_fixed_length
    	for(int c=0; c < Col; c++)
    	{
    		//for each row of loops_fixed_length create loop (vector)
    		std::vector<int> loop;
    		// fill loop with row elements of loops_fixed_length
    		for (int r = 0; r < Row; ++r)
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
        for (int j = 0; j < loop_list[i].size(); j++){
            cout << loop_list[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}