
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>


__global__ void add(int* a, int* c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		c[index] = a[index];
	}
}
void vec_sum(int num_vec) {
	std::vector<int> vec(num_vec); // Создаем вектор размером 1 000 000
	std::fill_n(vec.begin(), vec.size(), 5);
	int sum = 0;

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < vec.size(); i++) {
		sum += vec[i];
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;

	
	std::cout << "Vec: " << num_vec << std::endl << std::endl;
	
	std::cout << "CPU: " << std::endl;
	std::cout << "Sum: " << sum << std::endl;
	std::cout << "Time: " << diff.count() << " s" << std::endl << std::endl;



	int* d_a, * d_c;
	int size = vec.size() * sizeof(int);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, vec.data(), size, cudaMemcpyHostToDevice);

	start = std::chrono::high_resolution_clock::now();

	add << <1000, 1000 >> > (d_a, d_c, vec.size());

	cudaDeviceSynchronize();

	end = std::chrono::high_resolution_clock::now();
	diff = end - start;

	std::vector<int> result(vec.size());
	cudaMemcpy(result.data(), d_c, size, cudaMemcpyDeviceToHost);

	sum = 0;
	for (int i = 0; i < result.size(); i++) {
		sum += result[i];
	}

	std::cout << "GPU: " << std::endl;
	std::cout << "Sum: " << sum << std::endl;
	std::cout << "Time: " << diff.count() << " s" << std::endl;
	std::cout << "---------------------------------------------" << std::endl;

	cudaFree(d_a);
	cudaFree(d_c);

}

int main() {
	for (int i = 1000; i <= 1001000; i += 100000) {
		vec_sum(i);
	}

	return 0;
}



