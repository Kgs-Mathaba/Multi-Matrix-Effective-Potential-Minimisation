#include <iostream>

#include <stdio.h>      /* printf */
#include <time.h>       /* time_t, time, ctime */

using namespace std;
#include <ctime>
// Eigen section
#include <Eigen/Core>
 // Algebraic operations of dense matrices (inverse, eigenvalues, etc.)
#include <Eigen/Dense>


 
#define MATRIX_SIZE 100
 
/****************************
 * This program demonstrates the use of Eigen basic types
****************************/
 
int main( int argc, char** argv )
{
         // Solving equations 
         // We solve the equation A * x = b
         // Direct inversion is the most direct, but the amount of inverse operations is large.
 
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > A1;
    A1 = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
 
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > b1;
    b1 = Eigen::MatrixXd::Random( MATRIX_SIZE, 1 );
 
         Clock_t time_stt = clock(); // timing
         // Direct inversion
    Eigen::Matrix<double,MATRIX_SIZE,1> x = A1.inverse()*b1;
    cout <<"time use in normal inverse is " << 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms"<< endl;
    cout<<x<<endl;
         // QR decomposition colPivHouseholderQr()
    time_stt = clock();
    x = A1.colPivHouseholderQr().solve(b1);
    cout <<"time use in Qr decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;
         //QR decomposition fullPivHouseholderQr()
    time_stt = clock();
    x = A1.fullPivHouseholderQr().solve(b1);
    cout <<"time use in Qr decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;
         /* //llt decomposition requires matrix A positive definite
    time_stt = clock();
    x = A1.llt().solve(b1);
    cout <<"time use in llt decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;*/
         /*//ldlt decomposition requires matrix A to be positive or negative
    time_stt = clock();
    x = A1.ldlt().solve(b1);
    cout <<"time use in ldlt decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;*/
         //lu decomposition partialPivLu()
    time_stt = clock();
    x = A1.partialPivLu().solve(b1);
    cout <<"time use in lu decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;
         //lu decomposition (fullPivLu()
    time_stt = clock();
    x = A1.fullPivLu().solve(b1);
    cout <<"time use in lu decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;
 
    return 0;
 
}
