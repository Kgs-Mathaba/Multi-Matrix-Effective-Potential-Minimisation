#include <iostream>
#include <Eigen/Dense>
 
using namespace std;
using namespace Eigen;
 
int main()
{
   int n;
    n = 500;
    MatrixXf m = MatrixXf::Random(n, n);
    cout << "Here is the matrix m: " << endl << m <<endl;
    VectorXf b = VectorXf::Random(n);
    cout << "Here is the vector b: " << endl <<  b <<endl;
    VectorXf x = m.lu().solve(b);
 cout << "Here is the vector x: " << endl << x <<endl;
}
