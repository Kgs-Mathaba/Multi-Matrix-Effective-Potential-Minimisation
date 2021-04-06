// C++ program to check if an array is
// subarray of another array

#include <bits/stdc++.h>
using namespace std;

// Function to check if an array is
// subarray of another array
bool isSubArray(int A[], int B[], int n, int m)
{
	// Two pointers to traverse the arrays
	int i = 0, j = 0;

	// Traverse both arrays simultaneously
	while (i < n && j < m) {

		// If element matches
		// increment both pointers
		if (A[i] == B[j]) {

			i++;
			j++;

			// If array B is completely
			// traversed
			if (j == m)
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

// Driver Code
int main()
{
	int A[] = { 2, 1, 1, 2, 1, 1 };
	int n = sizeof(A) / sizeof(int);
	int B[] = { 1, 2, 1 };
	int m = sizeof(B) / sizeof(int);


	int logic[4][9] = {
    {0,1,8,8,8,8,8,1,1},
    {1,0,1,1,8,8,8,1,1},
    {8,1,0,1,8,8,8,8,1},
    {8,1,1,0,1,1,8,8,1}
	};


	int* loop_array_2 = logic[2];
	int o = sizeof(loop_array_2) / sizeof(int);

	if (isSubArray(loop_array_2, B, o, m))
		cout << "YES\n";
	else
		cout << "NO\n";

	for(int i=0; i<9; ++i){
		cout << loop_array_2[i];
	}

	return 0;
}
