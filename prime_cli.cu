#include <cuda_runtime.h> // For CUDA Runtime API functions
#include <device_launch_parameters.h> // For <<<...>>>
#include <stdio.h>
#include <stdlib.h> // For strtoll
#include <vector>
#include <cmath>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstring>

// --- Constants and Global Variables ---

#define THREADS_PER_BLOCK 256
// Size of the segment (chunk) to process on the GPU at a time.
// Adjust based on GPU memory. 8M of 'char' = 8MB.
#define SEGMENT_SIZE (1024 * 1024 * 8)

// Shared variables between the generator thread and the writer thread
std::vector<long long> primes_to_write_buffer; // Buffer to store found primes
std::mutex buffer_mutex;                     // Mutex to protect buffer access
std::condition_variable buffer_cv;           // Condition variable to notify the writer
std::atomic<bool> generation_done(false);    // Flag to indicate that generation is complete

// --- Helper Functions and CUDA Kernel ---

// Macro to check CUDA errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * @brief CUDA kernel to mark non-prime numbers in a segment.
 * (No changes from the previous version)
 */
__global__ void segmented_sieve_kernel(char *d_segment_is_prime,
                                       long long segment_start,
                                       long long segment_len,
                                       long long *d_base_primes,
                                       int num_base_primes)
{
    // Global thread index within the grid
    long long thread_idx_global = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over the numbers assigned to this thread using a grid-stride loop
    for (long long current_idx_in_segment = thread_idx_global;
         current_idx_in_segment < segment_len;
         current_idx_in_segment += (long long)gridDim.x * blockDim.x)
    {
        // Get the actual number corresponding to this segment index
        long long current_number = segment_start + current_idx_in_segment;

        // Mark multiples of the base primes
        for (int i = 0; i < num_base_primes; ++i) {
            long long p = d_base_primes[i];
            long long p_squared = p * p;

            if (p_squared > current_number) {
                break;
            }

            // Optimization: If the current number is less than p_squared and is a base prime, do not mark it
            // (We already know that base primes are prime)
            // Although the main logic already avoids this if current_number == p,
            // this check might be marginally faster for small numbers.

            // If the current number is divisible by the base prime p
            if (current_number % p == 0) {
                 // Ensure not to mark the base prime itself if it appears in the segment
                 // (This should not happen if segment_start > sqrt_limit, but for safety)
                 if (current_number != p) {
                    d_segment_is_prime[current_idx_in_segment] = 0; // Mark as non-prime
                 }
                break; // A single factor is enough to mark it as non-prime
            }
        }
    }
}

/**
 * @brief Computes primes up to 'limit' using the Sieve of Eratosthenes on the CPU.
 * (No changes from the previous version)
 */
void cpu_sieve(long long limit, std::vector<long long>& base_primes) {
    if (limit < 2) return;
    std::vector<char> is_prime(limit + 1, 1); // Initialize all as prime (1)
    is_prime[0] = is_prime[1] = 0; // 0 and 1 are not prime

    for (long long p = 2; p * p <= limit; ++p) {
        if (is_prime[p]) {
            for (long long i = p * p; i <= limit; i += p)
                is_prime[i] = 0; // Mark multiples as non-prime
        }
    }

    // Collect the primes
    base_primes.clear();
    base_primes.reserve(limit / (log(limit) > 1 ? log(limit) : 1)); // Estimate for reserve
    for (long long p = 2; p <= limit; ++p) {
        if (is_prime[p]) {
            base_primes.push_back(p);
        }
    }
}

/**
 * @brief Function executed by the writer thread to save primes to a file.
 * (No functional changes)
 */
void write_primes_to_file() {
    std::ofstream prime_file("primes.txt");
    if (!prime_file.is_open()) {
        fprintf(stderr, "Error: Could not open primes.txt for writing.\n");
        // We could add a mechanism to notify the main thread of the error
        return;
    }
    printf("Writer thread started. Writing to primes.txt...\n");

    std::vector<long long> local_buffer;
    local_buffer.reserve(200000); // Increase reserve if writing is a bottleneck

    while (true) {
        std::unique_lock<std::mutex> lock(buffer_mutex);
        buffer_cv.wait(lock, []{
            return generation_done.load() || !primes_to_write_buffer.empty();
        });

        if (!primes_to_write_buffer.empty()) {
            // Efficiently move the content
            local_buffer.insert(local_buffer.end(),
                                std::make_move_iterator(primes_to_write_buffer.begin()),
                                std::make_move_iterator(primes_to_write_buffer.end()));
            primes_to_write_buffer.clear(); // Clear the shared buffer
             // We could reduce capacity if it consumes too much memory:
             // primes_to_write_buffer.shrink_to_fit();
        }

        bool should_exit = generation_done.load() && primes_to_write_buffer.empty();
        lock.unlock(); // Release mutex BEFORE writing to file

        if (!local_buffer.empty()) {
            for (const auto& prime : local_buffer) {
                prime_file << prime << "\n";
                // Check for write errors (optional but recommended)
                // if (!prime_file) { fprintf(stderr, "Error writing to file!\n"); /* handle error */ }
            }
            local_buffer.clear(); // Clear local buffer after writing
            // local_buffer.shrink_to_fit(); // Optional: free memory
        }

        if (should_exit) {
            break;
        }
    }

    // Ensure to write any remaining data (though the above logic should cover it)
    if (!local_buffer.empty()) {
         printf("Writer thread: Writing final remaining data...\n");
        for (const auto& prime : local_buffer) {
            prime_file << prime << "\n";
        }
    }

    prime_file.close();
    printf("Writer thread: Writing completed.\n");
}

/**
 * @brief Generates primes using the GPU with segmented sieve.
 * (No functional changes)
 * @return float Total accumulated GPU time (in ms) measured by CUDA events.
 */
float generate_primes_gpu(long long upper_limit, const std::vector<long long>& base_primes) {
    if (upper_limit < 2) return 0.0f;

    long long sqrt_limit = static_cast<long long>(sqrt(upper_limit));

    // --- Add base primes to the write buffer ---
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        primes_to_write_buffer.insert(primes_to_write_buffer.end(), base_primes.begin(), base_primes.end());
    }
    buffer_cv.notify_one();

    // --- Prepare GPU ---
    long long *d_base_primes = nullptr;
    size_t base_primes_size_bytes = base_primes.size() * sizeof(long long);

    gpuErrchk(cudaMalloc(&d_base_primes, base_primes_size_bytes));
    // Use default stream 0 for the initial copy of base primes
    gpuErrchk(cudaMemcpy(d_base_primes, base_primes.data(), base_primes_size_bytes, cudaMemcpyHostToDevice));

    // --- Prepare Segment Buffers and CUDA Events ---
    char *h_segment = nullptr; // Use pinned memory for H->D async
    gpuErrchk(cudaMallocHost(&h_segment, SEGMENT_SIZE * sizeof(char)));
    // char *h_segment = new char[SEGMENT_SIZE]; // Alternative if pinned does not provide advantages
    char *d_segment = nullptr;
    gpuErrchk(cudaMalloc(&d_segment, SEGMENT_SIZE * sizeof(char)));

    cudaEvent_t start_event, stop_event;
    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    float total_gpu_time_ms = 0.0f;

    printf("Starting segmented processing on GPU...\n");

    long long segment_start = sqrt_limit + 1;
    if (segment_start < 2) segment_start = 2;

    // Create a CUDA stream for potential overlap (optional but good practice)
    cudaStream_t stream = 0; // Use default stream or create one: cudaStreamCreate(&stream);

    while (segment_start <= upper_limit) {
        long long segment_end = std::min(segment_start + SEGMENT_SIZE - 1, upper_limit);
        long long current_segment_len = segment_end - segment_start + 1;

        gpuErrchk(cudaEventRecord(start_event, stream)); // Mark start in the stream

        // 1. Initialize segment on host (CPU work)
        memset(h_segment, 1, current_segment_len * sizeof(char));

        // 2. Copy segment to device (H->D Async)
        gpuErrchk(cudaMemcpyAsync(d_segment, h_segment, current_segment_len * sizeof(char), cudaMemcpyHostToDevice, stream));

        // 3. Launch kernel (in the stream)
        int num_blocks = (current_segment_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        segmented_sieve_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_segment, segment_start, current_segment_len, d_base_primes, base_primes.size()
        );
        gpuErrchk(cudaGetLastError()); // Check for immediate launch error

        // 4. Copy result to host (D->H Async)
        // Note: Collection (step 5) depends on this copy. If using pinned memory
        // and wanting to overlap collection with the next iteration, double buffering
        // or more careful synchronization is needed. For simplicity, synchronize here.
        gpuErrchk(cudaMemcpyAsync(h_segment, d_segment, current_segment_len * sizeof(char), cudaMemcpyDeviceToHost, stream));

        gpuErrchk(cudaEventRecord(stop_event, stream)); // Mark end in the stream
        // Synchronize here to ensure D->H copy completes before processing h_segment
        // and before calculating event time.
        gpuErrchk(cudaEventSynchronize(stop_event));

        float segment_gpu_time_ms = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&segment_gpu_time_ms, start_event, stop_event));
        total_gpu_time_ms += segment_gpu_time_ms;

        // 5. Collect primes from the segment and add to buffer (CPU work)
        {
            std::vector<long long> segment_primes;
            segment_primes.reserve(current_segment_len / 10); // Estimate
            for (long long i = 0; i < current_segment_len; ++i) {
                if (h_segment[i] == 1) {
                    segment_primes.push_back(segment_start + i);
                }
            }
            // Add to shared buffer
            std::lock_guard<std::mutex> lock(buffer_mutex);
            primes_to_write_buffer.insert(primes_to_write_buffer.end(),
                                          std::make_move_iterator(segment_primes.begin()),
                                          std::make_move_iterator(segment_primes.end()));
        }
        buffer_cv.notify_one(); // Notify the writer

        segment_start = segment_end + 1;
    }

    printf("Segmented processing on GPU completed.\n");

    // --- Cleanup ---
    // if (stream != 0) gpuErrchk(cudaStreamDestroy(stream)); // If a stream was created
    gpuErrchk(cudaFreeHost(h_segment)); // Free pinned memory
    // delete[] h_segment; // If new char[] was used
    gpuErrchk(cudaEventDestroy(start_event));
    gpuErrchk(cudaEventDestroy(stop_event));
    gpuErrchk(cudaFree(d_segment));
    gpuErrchk(cudaFree(d_base_primes));

    generation_done.store(true);
    buffer_cv.notify_one(); // Final notification in case the writer is waiting

    return total_gpu_time_ms;
}

// --- Main Function ---

int main(int argc, char *argv[]) {
    // --- Measure Total Execution Time ---
    auto overall_start_time = std::chrono::high_resolution_clock::now();

    // --- Process Command-Line Arguments ---
    if (argc != 2) {
        // Print usage message to stderr
        fprintf(stderr, "Usage: %s <upper_limit>\n", argv[0]);
        fprintf(stderr, "  <upper_limit>: The non-negative integer up to which to calculate primes.\n");
        return 1; // Exit with error code
    }

    char *endptr;
    long long upper_limit = strtoll(argv[1], &endptr, 10); // Base 10

    // Check for conversion errors from strtoll
    if (endptr == argv[1] || *endptr != '\0') {
        fprintf(stderr, "Error: The upper limit '%s' is not a valid integer.\n", argv[1]);
        return 1;
    }
    // Check if the number is negative
    if (upper_limit < 0) {
         fprintf(stderr, "Error: The upper limit must be a non-negative number (%lld provided).\n", upper_limit);
         return 1;
    }
    printf("Upper limit set to: %lld\n", upper_limit);

    if (upper_limit < 2) {
        printf("There are no primes less than or equal to %lld.\n", upper_limit);
        // Create an empty file if expected
        std::ofstream prime_file("primes.txt");
        prime_file.close();
        // Measure and exit cleanly
         auto overall_end_time = std::chrono::high_resolution_clock::now();
         auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);
         printf("\n--- Performance Analysis ---\n");
         printf("Total Execution Time:         %lld ms\n", overall_duration.count());
         printf("---------------------------------\n");
        return 0;
    }

    // --- CPU Sieve Measurement ---
    auto cpu_sieve_start_time = std::chrono::high_resolution_clock::now();
    std::vector<long long> base_primes;
    long long sqrt_limit = static_cast<long long>(sqrt(upper_limit));
    cpu_sieve(sqrt_limit, base_primes);
    auto cpu_sieve_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_sieve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_sieve_end_time - cpu_sieve_start_time);
    printf("CPU Sieve: Found %zu base primes up to %lld.\n", base_primes.size(), sqrt_limit);

    // --- Start Writer Thread ---
    std::thread writer_thread(write_primes_to_file);

    // --- GPU Generation Measurement (Host-side) ---
    auto gpu_gen_host_start_time = std::chrono::high_resolution_clock::now();
    float total_gpu_event_time_ms = generate_primes_gpu(upper_limit, base_primes); // Execute generation on GPU
    auto gpu_gen_host_end_time = std::chrono::high_resolution_clock::now();
    auto gpu_gen_host_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_gen_host_end_time - gpu_gen_host_start_time);

    // --- Writer Thread Wait Measurement ---
    auto writer_join_start_time = std::chrono::high_resolution_clock::now();
    writer_thread.join(); // Wait for the writer thread to finish
    auto writer_join_end_time = std::chrono::high_resolution_clock::now();
    auto writer_join_duration = std::chrono::duration_cast<std::chrono::milliseconds>(writer_join_end_time - writer_join_start_time);

    // --- Total Time Measurement ---
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);

    // --- Print Performance Analysis Results ---
    printf("\n--- Performance Analysis ---\n");
    printf("CPU Sieve Time (Base Primes): %lld ms\n", cpu_sieve_duration.count());
    printf("GPU Management Time (Host):   %lld ms\n", gpu_gen_host_duration.count());
    printf("Total GPU Time (CUDA Events): %.2f ms\n", total_gpu_event_time_ms);
    printf("Writer Thread Wait Time:      %lld ms\n", writer_join_duration.count());
    printf("---------------------------------\n");
    printf("Total Execution Time:         %lld ms\n", overall_duration.count());
    printf("---------------------------------\n");
    printf("\nProgram finished. Primes have been saved to primes.txt\n");

    return 0; // Successful exit
}
