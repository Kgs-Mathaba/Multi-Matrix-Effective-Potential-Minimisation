
#include <Eigen/Core>
#include <iostream>
#include <LBFGS.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace LBFGSpp;
using namespace Eigen;
using namespace std;

double effect_pot(const VectorXd& x, VectorXd& grad)
{
    const int n = x.size();
    VectorXd loop(n);
    for(int i = 0; i < n; i++)
        loop[i] = i*i;

    double f = (x - loop).squaredNorm();
    grad.noalias() = 2.0 * (x - loop);
    return f;

    
}

int main()
{

ArrayXf a = ArrayXf::Random(10);
  a *= 2;
  cout << "a =" << endl 
       << a << endl;
  cout << "a.abs() =" << endl 
       << a.abs() << endl;
  cout << "a.abs().sqrt() =" << endl 
       << a.abs().sqrt() << endl;
 
  ArrayXf Eigen = ArrayXf::Random(10);
  cout << "Here is the array m:" << endl << Eigen << endl;
  cout << "Here is the sum of the array m:" << endl << Eigen.sum() << endl;
  cout << "Here is the size of the array m:" << endl << Eigen.size() << endl;














    const int n = 10;
    LBFGSParam<double> param;
    LBFGSSolver<double> solver(param);

    VectorXd x = VectorXd::Random(n);
    cout << "Initial vector x =" << endl 
       << x.transpose() << endl;
    double fx;
    int niter = solver.minimize(effect_pot, x, fx);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    return 0;
}
