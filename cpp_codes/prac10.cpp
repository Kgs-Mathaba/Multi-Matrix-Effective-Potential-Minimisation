
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

int n = 10;
cout << VectorXd::LinSpaced(Sequential,n ,-0.5,0.5)*sqrt() << endl;

return 0;
}
