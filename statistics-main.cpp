#include <limits.h>
#include <omp.h>
#include <stdlib.h>
#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using namespace std;

void s_avg(int arr[], int n) {
    long sum = 0L;
    int i;
    for (i = 0; i < n; i++) {
        sum = sum + arr[i];
    }
    cout << sum / long(n);
}

void p_avg(int arr[], int n) {
    long sum = 0L;
    int i;
#pragma omp parallel for reduction(+ : sum) num_threads(16)
    for (i = 0; i < n; i++) {
        sum = sum + arr[i];
    }
    cout << sum / long(n);
}

void s_sum(int arr[], int n) {
    long sum = 0L;
    int i;
    for (i = 0; i < n; i++) {
        sum = sum + arr[i];
    }
    cout << sum;
}

void p_sum(int arr[], int n) {
    long sum = 0L;
    int i;
#pragma omp parallel for reduction(+ : sum) num_threads(16)
    for (i = 0; i < n; i++) {
        sum = sum + arr[i];
    }
    cout << sum;
}

void s_max(int arr[], int n) {
    int max_val = INT_MIN;
    int i;
    for (i = 0; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    cout << max_val;
}

void p_max(int arr[], int n) {
    int max_val = INT_MIN;
    int i;
#pragma omp parallel for reduction(max : max_val) num_threads(16)
    for (i = 0; i < n; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    cout << max_val;
}

void s_min(int arr[], int n) {
    int min_val = INT_MAX;
    int i;
    for (i = 0; i < n; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    cout << min_val;
}

void p_min(int arr[], int n) {
    int min_val = INT_MAX;
    int i;
#pragma omp parallel for reduction(min : min_val) num_threads(16)
    for (i = 0; i < n; i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    cout << min_val;
}

std::string bench_traverse(std::function<void()> traverse_fn) {
    auto start = high_resolution_clock::now();
    traverse_fn();
    cout << " (";
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    return std::to_string(duration.count());
}

int main(int argc, const char **argv) {
    if (argc < 3) {
        std::cout << "Specify array length and maximum random value\n";
        return 1;
    }
    int *a, n, rand_max;
    n = stoi(argv[1]);
    rand_max = stoi(argv[2]);
    a = new int[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand() % rand_max;
    }

    cout << "Generated random array of length " << n << " with elements between 0 to " << rand_max << "\n\n";

    cout << "Given array is:\n";
    for (int i = 0; i < n; i++) {
        cout << a[i] << ", ";
    }
    cout << "\n\n";

    omp_set_num_threads(16);

    std::cout << "Sequential Min: " << bench_traverse([&] { s_min(a, n); }) << "ms)\n";
    std::cout << "Parallel (16) Min: " << bench_traverse([&] { p_min(a, n); }) << "ms)\n\n";

    std::cout << "Sequential Max: " << bench_traverse([&] { s_max(a, n); }) << "ms)\n";
    std::cout << "Parallel (16) Max: " << bench_traverse([&] { p_max(a, n); }) << "ms)\n\n";

    std::cout << "Sequential Sum: " << bench_traverse([&] { s_sum(a, n); }) << "ms)\n";
    std::cout << "Parallel (16) Sum: " << bench_traverse([&] { p_sum(a, n); }) << "ms)\n\n";

    std::cout << "Sequential Average: " << bench_traverse([&] { s_avg(a, n); }) << "ms)\n";
    std::cout << "Parallel (16) Average: " << bench_traverse([&] { p_avg(a, n); }) << "ms)\n";

    delete[] a;
    return 0;
}
