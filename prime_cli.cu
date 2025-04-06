#include <cuda_runtime.h> // Para funciones CUDA Runtime API
#include <device_launch_parameters.h> // Para <<<...>>>
#include <stdio.h>
#include <stdlib.h> // Para strtoll
#include <vector>
#include <cmath>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm> // Para std::min, std::max
#include <chrono>    // Para medir el tiempo
#include <cstring>   // Para memset

// --- Constantes y Variables Globales ---

#define THREADS_PER_BLOCK 256
// Tamaño del segmento (chunk) a procesar en la GPU a la vez.
// Ajustar según la memoria de la GPU. 8M de 'char' = 8MB.
#define SEGMENT_SIZE (1024 * 1024 * 8)

// Variables compartidas entre el hilo generador y el escritor
std::vector<long long> primes_to_write_buffer; // Buffer para almacenar primos encontrados
std::mutex buffer_mutex;                     // Mutex para proteger el acceso al buffer
std::condition_variable buffer_cv;           // Variable de condición para notificar al escritor
std::atomic<bool> generation_done(false);    // Flag para indicar que la generación ha terminado

// --- Funciones Auxiliares y Kernel CUDA ---

// Macro para chequear errores de CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * @brief Kernel CUDA para marcar números no primos en un segmento.
 * (Sin cambios respecto a la versión anterior)
 */
__global__ void segmented_sieve_kernel(char *d_segment_is_prime,
                                       long long segment_start,
                                       long long segment_len,
                                       long long *d_base_primes,
                                       int num_base_primes)
{
    // Índice global del hilo dentro de la grid
    long long thread_idx_global = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Iterar sobre los números asignados a este hilo usando un grid-stride loop
    for (long long current_idx_in_segment = thread_idx_global;
         current_idx_in_segment < segment_len;
         current_idx_in_segment += (long long)gridDim.x * blockDim.x)
    {
        // Obtener el número real correspondiente a este índice del segmento
        long long current_number = segment_start + current_idx_in_segment;

        // Marcar múltiplos de los primos base
        for (int i = 0; i < num_base_primes; ++i) {
            long long p = d_base_primes[i];
            long long p_squared = p * p;

            if (p_squared > current_number) {
                break;
            }

            // Optimización: Si el número actual es menor que p_squared y es primo base, no lo marcamos
            // (Ya sabemos que los primos base son primos)
            // Aunque la lógica principal ya evita esto si current_number == p,
            // esta comprobación puede ser marginalmente más rápida para números pequeños.
            // Sin embargo, la comprobación `if (current_number % p == 0)` es la clave.

            // Si el número actual es divisible por el primo base p
            if (current_number % p == 0) {
                 // Asegurarse de no marcar el propio primo base si aparece en el segmento
                 // (Esto no debería pasar si segment_start > sqrt_limit, pero por seguridad)
                 if (current_number != p) {
                    d_segment_is_prime[current_idx_in_segment] = 0; // Marcar como no primo
                 }
                break; // Un solo factor es suficiente para marcarlo como no primo
            }
        }
    }
}

/**
 * @brief Calcula primos hasta 'limit' usando la Criba de Eratóstenes en la CPU.
 * (Sin cambios respecto a la versión anterior)
 */
void cpu_sieve(long long limit, std::vector<long long>& base_primes) {
    if (limit < 2) return;
    std::vector<char> is_prime(limit + 1, 1); // Inicializar todo a primo (1)
    is_prime[0] = is_prime[1] = 0; // 0 y 1 no son primos

    for (long long p = 2; p * p <= limit; ++p) {
        if (is_prime[p]) {
            for (long long i = p * p; i <= limit; i += p)
                is_prime[i] = 0; // Marcar múltiplos como no primos
        }
    }

    // Recolectar los primos
    base_primes.clear();
    base_primes.reserve(limit / (log(limit) > 1 ? log(limit) : 1)); // Estimación para reserva
    for (long long p = 2; p <= limit; ++p) {
        if (is_prime[p]) {
            base_primes.push_back(p);
        }
    }
}

/**
 * @brief Función ejecutada por el hilo escritor para guardar los primos en un archivo.
 * (Sin cambios funcionales)
 */
void write_primes_to_file() {
    std::ofstream prime_file("primes.txt");
    if (!prime_file.is_open()) {
        fprintf(stderr, "Error: No se pudo abrir el archivo primes.txt para escritura.\n");
        // Podríamos añadir un mecanismo para notificar al hilo principal del error
        return;
    }
    printf("Hilo escritor iniciado. Escribiendo en primes.txt...\n");

    std::vector<long long> local_buffer;
    local_buffer.reserve(200000); // Aumentar reserva si la escritura es cuello de botella

    while (true) {
        std::unique_lock<std::mutex> lock(buffer_mutex);
        buffer_cv.wait(lock, []{
            return generation_done.load() || !primes_to_write_buffer.empty();
        });

        if (!primes_to_write_buffer.empty()) {
            // Mover eficientemente el contenido
            local_buffer.insert(local_buffer.end(),
                                std::make_move_iterator(primes_to_write_buffer.begin()),
                                std::make_move_iterator(primes_to_write_buffer.end()));
            primes_to_write_buffer.clear(); // Limpiar el buffer compartido
             // Podríamos reducir la capacidad si consume mucha memoria:
             // primes_to_write_buffer.shrink_to_fit();
        }

        bool should_exit = generation_done.load() && primes_to_write_buffer.empty();
        lock.unlock(); // Liberar mutex ANTES de escribir en archivo

        if (!local_buffer.empty()) {
            for (const auto& prime : local_buffer) {
                prime_file << prime << "\n";
                // Comprobar errores de escritura (opcional pero recomendado)
                // if (!prime_file) { fprintf(stderr, "Error escribiendo en archivo!\n"); /* manejar error */ }
            }
            local_buffer.clear(); // Limpiar buffer local después de escribir
            // local_buffer.shrink_to_fit(); // Opcional: liberar memoria
        }

        if (should_exit) {
            break;
        }
    }

    // Asegurarse de escribir cualquier remanente (aunque la lógica anterior debería cubrirlo)
    if (!local_buffer.empty()) {
         printf("Hilo escritor: Escribiendo remanente final...\n");
        for (const auto& prime : local_buffer) {
            prime_file << prime << "\n";
        }
    }

    prime_file.close();
    printf("Hilo escritor: Escritura completada.\n");
}

/**
 * @brief Genera primos usando la GPU con la criba segmentada.
 * (Sin cambios funcionales)
 * @return float Tiempo total acumulado en GPU (en ms) medido por eventos CUDA.
 */
float generate_primes_gpu(long long upper_limit, const std::vector<long long>& base_primes) {
    if (upper_limit < 2) return 0.0f;

    long long sqrt_limit = static_cast<long long>(sqrt(upper_limit));

    // --- Añadir primos base al buffer de escritura ---
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        primes_to_write_buffer.insert(primes_to_write_buffer.end(), base_primes.begin(), base_primes.end());
    }
    buffer_cv.notify_one();

    // --- Preparar GPU ---
    long long *d_base_primes = nullptr;
    size_t base_primes_size_bytes = base_primes.size() * sizeof(long long);

    gpuErrchk(cudaMalloc(&d_base_primes, base_primes_size_bytes));
    // Usar stream 0 por defecto para la copia inicial de primos base
    gpuErrchk(cudaMemcpy(d_base_primes, base_primes.data(), base_primes_size_bytes, cudaMemcpyHostToDevice));

    // --- Preparar Buffers de Segmento y Eventos CUDA ---
    char *h_segment = nullptr; // Usar memoria pinned para H->D async
    gpuErrchk(cudaMallocHost(&h_segment, SEGMENT_SIZE * sizeof(char)));
    // char *h_segment = new char[SEGMENT_SIZE]; // Alternativa si pinned no da ventajas
    char *d_segment = nullptr;
    gpuErrchk(cudaMalloc(&d_segment, SEGMENT_SIZE * sizeof(char)));

    cudaEvent_t start_event, stop_event;
    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
    float total_gpu_time_ms = 0.0f;

    printf("Iniciando procesamiento segmentado en GPU...\n");

    long long segment_start = sqrt_limit + 1;
    if (segment_start < 2) segment_start = 2;

    // Crear un stream CUDA para solapamiento potencial (opcional pero buena práctica)
    cudaStream_t stream = 0; // Usar stream por defecto o crear uno: cudaStreamCreate(&stream);

    while (segment_start <= upper_limit) {
        long long segment_end = std::min(segment_start + SEGMENT_SIZE - 1, upper_limit);
        long long current_segment_len = segment_end - segment_start + 1;

        gpuErrchk(cudaEventRecord(start_event, stream)); // Marcar inicio en el stream

        // 1. Inicializar segmento en host (CPU work)
        memset(h_segment, 1, current_segment_len * sizeof(char));

        // 2. Copiar segmento a device (H->D Async)
        gpuErrchk(cudaMemcpyAsync(d_segment, h_segment, current_segment_len * sizeof(char), cudaMemcpyHostToDevice, stream));

        // 3. Lanzar kernel (en el stream)
        int num_blocks = (current_segment_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        segmented_sieve_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_segment, segment_start, current_segment_len, d_base_primes, base_primes.size()
        );
        gpuErrchk(cudaGetLastError()); // Comprobar error de lanzamiento inmediato

        // 4. Copiar resultado a host (D->H Async)
        // Nota: La recolección (paso 5) depende de esta copia. Si usamos pinned memory
        // y queremos solapar la recolección con la siguiente iteración, se necesita
        // doble buffer o sincronización más cuidadosa. Por simplicidad, sincronizamos aquí.
        gpuErrchk(cudaMemcpyAsync(h_segment, d_segment, current_segment_len * sizeof(char), cudaMemcpyDeviceToHost, stream));

        gpuErrchk(cudaEventRecord(stop_event, stream)); // Marcar fin en el stream
        // Sincronizar aquí asegura que la copia D->H se complete antes de procesar h_segment
        // y antes de calcular el tiempo del evento.
        gpuErrchk(cudaEventSynchronize(stop_event));

        float segment_gpu_time_ms = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&segment_gpu_time_ms, start_event, stop_event));
        total_gpu_time_ms += segment_gpu_time_ms;

        // 5. Recolectar primos del segmento y añadir al buffer (CPU work)
        {
            std::vector<long long> segment_primes;
            segment_primes.reserve(current_segment_len / 10); // Estimación
            for (long long i = 0; i < current_segment_len; ++i) {
                if (h_segment[i] == 1) {
                    segment_primes.push_back(segment_start + i);
                }
            }
            // Añadir al buffer compartido
            std::lock_guard<std::mutex> lock(buffer_mutex);
            primes_to_write_buffer.insert(primes_to_write_buffer.end(),
                                          std::make_move_iterator(segment_primes.begin()),
                                          std::make_move_iterator(segment_primes.end()));
        }
        buffer_cv.notify_one(); // Notificar al escritor

        segment_start = segment_end + 1;
    }

    printf("Procesamiento segmentado en GPU completado.\n");

    // --- Limpieza ---
    // if (stream != 0) gpuErrchk(cudaStreamDestroy(stream)); // Si se creó un stream
    gpuErrchk(cudaFreeHost(h_segment)); // Liberar memoria pinned
    // delete[] h_segment; // Si se usó new char[]
    gpuErrchk(cudaEventDestroy(start_event));
    gpuErrchk(cudaEventDestroy(stop_event));
    gpuErrchk(cudaFree(d_segment));
    gpuErrchk(cudaFree(d_base_primes));

    generation_done.store(true);
    buffer_cv.notify_one(); // Última notificación por si el escritor está esperando

    return total_gpu_time_ms;
}

// --- Función Principal ---

int main(int argc, char *argv[]) {
    // --- Medición de Tiempo Total ---
    auto overall_start_time = std::chrono::high_resolution_clock::now();

    // --- Procesar Argumentos de Línea de Comandos ---
    if (argc != 2) {
        // Imprimir mensaje de uso en stderr
        fprintf(stderr, "Uso: %s <limite_superior>\n", argv[0]);
        fprintf(stderr, "  <limite_superior>: El número entero (no negativo) hasta el cual calcular primos.\n");
        return 1; // Salir con código de error
    }

    char *endptr;
    long long upper_limit = strtoll(argv[1], &endptr, 10); // Base 10

    // Verificar errores de conversión de strtoll
    if (endptr == argv[1] || *endptr != '\0') {
        fprintf(stderr, "Error: El límite superior '%s' no es un número entero válido.\n", argv[1]);
        return 1;
    }
    // Verificar si el número es negativo
    if (upper_limit < 0) {
         fprintf(stderr, "Error: El límite superior debe ser un número no negativo (%lld proporcionado).\n", upper_limit);
         return 1;
    }
    printf("Límite superior establecido en: %lld\n", upper_limit);


    if (upper_limit < 2) {
        printf("No hay primos menores o iguales a %lld.\n", upper_limit);
        // Crear archivo vacío igualmente si se espera
        std::ofstream prime_file("primes.txt");
        prime_file.close();
        // Medir y salir limpiamente
         auto overall_end_time = std::chrono::high_resolution_clock::now();
         auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);
         printf("\n--- Análisis de Rendimiento ---\n");
         printf("Tiempo Total Ejecución:         %lld ms\n", overall_duration.count());
         printf("---------------------------------\n");
        return 0;
    }

    // --- Medición CPU Sieve ---
    auto cpu_sieve_start_time = std::chrono::high_resolution_clock::now();
    std::vector<long long> base_primes;
    long long sqrt_limit = static_cast<long long>(sqrt(upper_limit));
    cpu_sieve(sqrt_limit, base_primes);
    auto cpu_sieve_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_sieve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_sieve_end_time - cpu_sieve_start_time);
    printf("CPU Sieve: Encontrados %zu primos base hasta %lld.\n", base_primes.size(), sqrt_limit);


    // --- Iniciar Hilo Escritor ---
    std::thread writer_thread(write_primes_to_file);

    // --- Medición Generación GPU (Host-side) ---
    auto gpu_gen_host_start_time = std::chrono::high_resolution_clock::now();
    float total_gpu_event_time_ms = generate_primes_gpu(upper_limit, base_primes); // Ejecutar generación en GPU
    auto gpu_gen_host_end_time = std::chrono::high_resolution_clock::now();
    auto gpu_gen_host_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_gen_host_end_time - gpu_gen_host_start_time);


    // --- Medición Espera Hilo Escritor ---
    auto writer_join_start_time = std::chrono::high_resolution_clock::now();
    writer_thread.join(); // Esperar a que el hilo escritor termine
    auto writer_join_end_time = std::chrono::high_resolution_clock::now();
    auto writer_join_duration = std::chrono::duration_cast<std::chrono::milliseconds>(writer_join_end_time - writer_join_start_time);


    // --- Medición Tiempo Total ---
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end_time - overall_start_time);

    // --- Imprimir Resultados del Análisis de Velocidad ---
    printf("\n--- Análisis de Rendimiento ---\n");
    printf("Tiempo Criba CPU (Primos Base): %lld ms\n", cpu_sieve_duration.count());
    printf("Tiempo Gestión GPU (Host):      %lld ms\n", gpu_gen_host_duration.count());
    printf("Tiempo Total GPU (Eventos CUDA): %.2f ms\n", total_gpu_event_time_ms);
    printf("Tiempo Espera Hilo Escritor:    %lld ms\n", writer_join_duration.count());
    printf("---------------------------------\n");
    printf("Tiempo Total Ejecución:         %lld ms\n", overall_duration.count());
    printf("---------------------------------\n");
    printf("\nPrograma finalizado. Los primos se han guardado en primes.txt\n");

    return 0; // Salida exitosa
}
