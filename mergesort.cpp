#include <omp.h>
#include <stdlib.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <thread>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using namespace std;

// ---------- Bubble Sort ----------
void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

void s_bubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void p_bubble(int *a, int n) {
    for (int i = 0; i < n; i++) {
        int first = i % 2;
#pragma omp parallel for shared(a, first)
        for (int j = first; j < n - 1; j += 2) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// ---------- Merge Sort ----------
void merge(int *a, int *temp, int i1, int j1, int i2, int j2) {
    int i = i1, j = i2, k = 0;

    while (i <= j1 && j <= j2) {
        if (a[i] < a[j]) {
            temp[k++] = a[i++];
        } else {
            temp[k++] = a[j++];
        }
    }
    while (i <= j1) {
        temp[k++] = a[i++];
    }
    while (j <= j2) {
        temp[k++] = a[j++];
    }
    for (i = i1, j = 0; i <= j2; i++, j++) {
        a[i] = temp[j];
    }
}

void s_mergesort(int *a, int *temp, int i, int j) {
    int mid;
    if (i < j) {
        mid = (i + j) / 2;
        s_mergesort(a, temp, i, mid);
        s_mergesort(a, temp, mid + 1, j);
        merge(a, temp, i, mid, mid + 1, j);
    }
}

void p_mergesort(int *a, int *temp, int i, int j) {
    int mid;
    if (i < j) {
        if ((j - i) > 1000) {
            mid = (i + j) / 2;

#pragma omp task shared(a, temp)
            p_mergesort(a, temp, i, mid);
#pragma omp task shared(a, temp)
            p_mergesort(a, temp, mid + 1, j);

#pragma omp taskwait
            merge(a, temp, i, mid, mid + 1, j);
        } else {
            s_mergesort(a, temp, i, j);
        }
    }
}

void parallel_mergesort(int *a, int *temp, int i, int j) {
#pragma omp parallel num_threads(thread::hardware_concurrency())
    {
#pragma omp single
        p_mergesort(a, temp, i, j);
    }
}

// ---------- Helper Functions ----------
std::string bench_traverse(std::function<void()> traverse_fn) {
    auto start = high_resolution_clock::now();
    traverse_fn();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    return std::to_string(duration.count());
}

void print_array(int *a, int n) {
    int limit = min(n, 20);
    for (int i = 0; i < limit; i++) {
        cout << a[i] << ", ";
    }
    cout << "... (showing first " << limit << " elements)\n";
}

// ---------- Main ----------
int main(int argc, const char **argv) {
    if (argc < 3) {
        std::cout << "Specify array length and maximum random value\n";
        return 1;
    }
    int *a, *b, *temp, n, rand_max;
    n = stoi(argv[1]);
    rand_max = stoi(argv[2]);
    a = new int[n];
    b = new int[n];
    temp = new int[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand() % rand_max;
    }
    copy(a, a + n, b);

    cout << "Generated random array of length " << n << " with elements between 0 to " << rand_max << "\n\n";

    cout << "Choose Sorting Algorithm:\n";
    cout << "1. Sequential Bubble Sort\n";
    cout << "2. Parallel Bubble Sort\n";
    cout << "3. Sequential Merge Sort\n";
    cout << "4. Parallel Merge Sort\n";
    cout << "Enter choice (1-4): ";
    int choice;
    cin >> choice;
    cout << "\n";

    switch (choice) {
        case 1:
            cout << "Sequential Bubble sort time: " << bench_traverse([&] { s_bubble(a, n); }) << " ms\n";
            print_array(a, n);
            break;

        case 2:
            cout << "Parallel Bubble sort time: " << bench_traverse([&] { p_bubble(b, n); }) << " ms\n";
            print_array(b, n);
            break;

        case 3:
            cout << "Sequential Merge sort time: " << bench_traverse([&] { s_mergesort(a, temp, 0, n - 1); }) << " ms\n";
            print_array(a, n);
            break;

        case 4:
            cout << "Parallel Merge sort time: " << bench_traverse([&] { parallel_mergesort(b, temp, 0, n - 1); }) << " ms\n";
            print_array(b, n);
            break;

        default:
            cout << "Invalid choice!\n";
    }

    delete[] a;
    delete[] b;
    delete[] temp;
    return 0;
}
