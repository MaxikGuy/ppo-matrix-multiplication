#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <immintrin.h>
#include <omp.h>

#define MKL
#ifdef MKL
#include "mkl.h"
#endif

using namespace std;

void generation(double * mat, size_t size)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
	for (size_t i = 0; i < size * size; i++)
		mat[i] = uniform_distance(gen);
}

void matrix_mult(const double* a, const double* b, double* res, size_t size)
{
    // Initialization of result
    #pragma omp parallel for
    for (size_t i = 0; i < size * size; i++) {
        res[i] = 0.0;
    }

    // Transpose matrix b to improve memory access
	// Significantly reduces the number of cache misses, because data will now be read line by line
    double* b_transposed = new double[size * size];
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            b_transposed[j * size + i] = b[i * size + j];
        }
    }

    // Block size for dividing the matrix into blocks
	// Blocks improve data locality by minimizing cache downloads.
    const size_t block_size = 64; // Choose the optimal size for your system
								  // For my PC (Intel i7-4770K 3.50GHz) best_block_size = 64

    // Main loop with blocks
	// The matrices are split into a block of block_size x block_size size.
	// This reduces the amount of data loaded into the cache in each cycle and increases the efficiency of cache usage.
    #pragma omp parallel for collapse(2) // parallelizes two levels of cycles (bi and bj), which allows you to use more threads at once.
    for (size_t bi = 0; bi < size; bi += block_size) {
        for (size_t bj = 0; bj < size; bj += block_size) {
            for (size_t bk = 0; bk < size; bk += block_size) {
				
				// The boundaries of the current block are determined so as not to go beyond the matrices when processing blocks at the borders.
				size_t i_end = std::min(bi + block_size, size);
				size_t j_end = std::min(bj + block_size, size);
				size_t k_end = std::min(bk + block_size, size);
                
				for (size_t i = bi; i < i_end; i++) {
                    for (size_t j = bj; j < j_end; j++) {
                        // Using AVX instructions allows you to perform 4 multiplication and addition operations in one processor command.
						__m256d sum = _mm256_set1_pd(0.0); // For storing partial sum
                        for (size_t k = bk; k < k_end; k += 4) { // "k += 4" because the AVX register processes 4 numbers at the same time.
                            // Loading data from matrices a and b_transposed
							// The FMA operation combines multiplication and addition into a single instruction, reducing the number of processor instructions and minimizing delays.
                            __m256d vec_a = _mm256_loadu_pd(&a[i * size + k]);
                            __m256d vec_b = _mm256_loadu_pd(&b_transposed[j * size + k]);
                            sum = _mm256_fmadd_pd(vec_a, vec_b, sum); // Fused multiply-add (FMA)
                        }
                        // Saving a partial sum
						// This allows you to process 4 items at a time, minimizing the need for multiple res calls.
                        double buffer[4];
                        _mm256_storeu_pd(buffer, sum);
                        res[i * size + j] += buffer[0] + buffer[1] + buffer[2] + buffer[3];
                    }
                }
            }
        }
    }

    delete[] b_transposed;
}

int main()
{
	double *mat, *mat_mkl, *a, *b, *a_mkl, *b_mkl;
	size_t size = 1000;
	chrono::time_point<chrono::system_clock> start, end;

	mat = new double[size * size];
	a = new double[size * size];
	b = new double[size * size];
	generation(a, size);
	generation(b, size);
	memset(mat, 0, size*size * sizeof(double));

#ifdef MKL     
    mat_mkl = new double[size * size];
	a_mkl = new double[size * size];
	b_mkl = new double[size * size];
	memcpy(a_mkl, a, sizeof(double)*size*size);
    memcpy(b_mkl, b, sizeof(double)*size*size);
	memset(mat_mkl, 0, size*size * sizeof(double));
#endif

	start = chrono::system_clock::now();
	matrix_mult(a, b, mat, size);
	end = chrono::system_clock::now();
    
   
	int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time: " << elapsed_seconds/1000.0 << " sec" << endl;

#ifdef MKL 
	start = chrono::system_clock::now();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
    end = chrono::system_clock::now();
    
    elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
		(end - start).count();
	cout << "Total time mkl: " << elapsed_seconds/1000.0 << " sec" << endl;
     
    int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if(abs(mat[i] - mat_mkl[i]) > size*1e-14){
		    flag = 1;
        }
    if (flag)
        cout << "fail" << endl;
    else
        cout << "correct" << endl; 
    
    delete (a_mkl);
    delete (b_mkl);
    delete (mat_mkl);
#endif

    delete (a);
    delete (b);
    delete (mat);

	//system("pause");
	
	return 0;
}
