#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;
using namespace std;

int main()
{
    chrono::steady_clock sc;   // create an object of `steady_clock` class
    int n;
    n = 5000;
    MatrixXf m = MatrixXf::Random(n, n);
    VectorXf b = VectorXf::Random(n);
    auto start = sc.now();     // start timer
    VectorXf x = m.lu().solve(b);
    auto end = sc.now();
    // measure time span between start & end
    auto time_span = static_cast<chrono::duration<double>>(end - start);
    cout << "Operation took: " << time_span.count() << " seconds !!!";
}
