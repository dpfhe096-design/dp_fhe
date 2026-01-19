#define PROFILE // turns on the reporting of timing results

#include "openfhe.h"
#include "util/crypto.h"    
#include "util/data_prep.h" 
#include "util/format.h"    
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <random>
#include <numeric>
#include <omp.h>
#include <thread>
#include <atomic>

#include "utils/serial.h"
#include "cryptocontext-ser.h"
#include "ciphertext-ser.h"
#include "key/key-ser.h"

#include <mutex>      // For std::mutex
#include <iomanip>    // For std::setprecision

using namespace lbcrypto;

double getCurrentMemoryMB() {
    std::ifstream statm("/proc/self/status");
    std::string line;
    while (std::getline(statm, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line);
            std::string key, unit;
            long value;
            iss >> key >> value >> unit; // VmRSS: <value> kB
            return value / 1024.0;       // convert KB → MB
        }
    }
    return 0.0;
}

// --- Global RAM Tracker ---
double max_ram_mb = 0.0;
std::mutex ram_mutex;

/**
 * @brief Checks current RAM, updates the global max, and prints the status.
 * @param log_message A description of
 * @return The current RAM usage in MB.
 */
double checkAndUpdateMaxRAM(const std::string& log_message) {
    double current_ram = getCurrentMemoryMB();
    {
        std::lock_guard<std::mutex> lock(ram_mutex);
        if (current_ram > max_ram_mb) {
            max_ram_mb = current_ram;
        }
    }

    // Print with fixed precision
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[RAM] " << log_message << ": "
              << current_ram << " MB"
              << " (Current Max: " << max_ram_mb << " MB)" << std::endl;
    std::cout << std::defaultfloat; // Reset precision formatting

    return current_ram;
}


// --- Decrypt and Display full weight at iteration iter ---
void PrintWeights(
    const CryptoContext<DCRTPoly>& cc, 
    const PrivateKey<DCRTPoly>& sk,
    const Ciphertext<DCRTPoly>& ct,
    size_t length,
    int iter) { 

    Plaintext pl_weights;
    cc->Decrypt(sk, ct, &pl_weights);
    pl_weights->SetLength(length);

    auto weights_vec = pl_weights->GetCKKSPackedValue();

    std::cout << "\n--- " << "weights at iteration= " << iter << " ---" << std::endl;
    std::cout << "(" ;
    for (size_t j = 0; j < weights_vec.size(); ++j) {
        std::cout << weights_vec[j].real();
        if (j < weights_vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
}


// --- Helper function to read CSV of indices ---
std::vector<std::vector<size_t>> readCSVIndices(const std::string& filename) {
    std::vector<std::vector<size_t>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open index file: " << filename << std::endl;
        throw std::runtime_error("Could not open index file: " + filename);
    }
    std::cout << "Successfully opened index file: " << filename << std::endl;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue; // Skip empty lines
        
        std::vector<size_t> row;
        std::stringstream ss(line);
        std::string field;

        while (std::getline(ss, field, ',')) {
            try {
                // Use stoull (string to unsigned long long) for size_t
                row.push_back(std::stoull(field)); 
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: Error parsing index value '" << field << "': " << e.what() << std::endl;
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;
}

// --- PASTE YOUR DP_GD_no_clipping CLASS HERE ---
// (The class definition and its member functions: calculate_gradient and fit)

// code for training DP-GD with clipping
class DP_GD_no_clipping {
public:
    // Parameters:
    double learning_rate;
    int T;
    double C;
    double loss;
    double theta;
    double lambda;

    // Logs
    std::vector<Ciphertext<DCRTPoly>> weights_log;
    std::vector<Ciphertext<DCRTPoly>> loss_log;

    DP_GD_no_clipping(
        double learning_rate,
        int T,
        double C,
        double loss,
        double theta,
        double lambda) : learning_rate(learning_rate),
                         T(T),
                         C(C),
                         loss(loss),
                         theta(theta),
                         lambda(lambda){};

    // Coefficients for polynomial approximations
    std::vector<double> sigmoid_coeffs = {
        // sigmoid coeff for compas
        // 0.5, 0.21763506, 0, -0.00832304, 0, 0.00017120, 0, -0.00000126
        // sigmoid coeff for mnist
        // 0.5, 0.15266035, 0, -0.00221240, 0, 0.00001614, 0, -0.00000005
        // sigmoid coeff for adult
        // 0.5, 0.21486557, 0, -0.00804063, 0, 0.00017484, 0, -0.00000176, 0, 0.00000001
        // sigmoid coeff for credit
        0.5, 0.19282043, 0, -0.00514345, 0, 0.00007417, 0, -0.00000048
    };


    std::vector<double> p_kappa_coeffs = {
        // sigmoid coeff for compas (deg 4)
        // 159.18274753, -399.07926696, 286.16850136, -77.44645077, 7.03915871,
        // sigmoid coeff for mnist (deg 4)
        // 473.48344513, -1144.60809557, 802.83432319, -214.14083226, 19.26469781
        // sigmoid coeff for adult (deg 5)
        // 263.00923284, -600.98138942,435.6657172,-133.65452022,17.71317527,-0.79815745,
        // sigmoid coeff for credit (deg 4)
        389.51488958, -992.19946114, 715.77120495, -194.33403444, 17.69867058
    };




    std::vector<double> inverse_interval = {
        // interval for mnist
        0.0005, 5};

    // Merged and optimized function to compute the entire gradient in one pass.
    Ciphertext<DCRTPoly> calculate_gradient(
        const CryptoContext<DCRTPoly>& cc,
        size_t dim_n,
        size_t dim_m,
        const std::vector<Ciphertext<DCRTPoly>>& input_X,
        const std::vector<Ciphertext<DCRTPoly>>& Y,
        const Ciphertext<DCRTPoly>& noise,
        const Ciphertext<DCRTPoly>& weights,
        PublicKey<DCRTPoly>& publicKey);

    // define function for training the model
    Ciphertext<DCRTPoly> fit(
        const CryptoContext<DCRTPoly>& cc,
        size_t dim_n,
        size_t dim_m,
        // const std::vector<Ciphertext<DCRTPoly>>& X,
        // const std::vector<Ciphertext<DCRTPoly>>& Y,
        const std::string& x_data_dir, // MODIFIED
        const std::string& y_data_dir, // MODIFIED
        const std::vector<Ciphertext<DCRTPoly>>& noise_list,
        uint32_t T, 
        PublicKey<DCRTPoly>& publicKey,
        size_t batch_size,
        const std::string& weights_output_dir
    );
};


Ciphertext<DCRTPoly> DP_GD_no_clipping::calculate_gradient(
    const CryptoContext<DCRTPoly>& cc,
    size_t dim_n,
    size_t dim_m,
    const std::vector<Ciphertext<DCRTPoly>>& input_X,
    const std::vector<Ciphertext<DCRTPoly>>& Y,
    const Ciphertext<DCRTPoly>& noise,
    const Ciphertext<DCRTPoly>& weights,
    PublicKey<DCRTPoly>& publicKey) {
    // checkAndUpdateMaxRAM("\t\tGradCalc / Start");


    
    
// // ------------------------------------------------------------
// // parallel method 1
// // ------------------------------------------------------------
// ------------Path A: Gradient Calculation -----------------------
    // // --- 1. Linear Hypothesis: z = <x, w>
    // std::vector<Ciphertext<DCRTPoly>> z_vec = EncryptedMatrixVectorMultiply(cc,
    //                                                                          input_X,
    //                                                                          weights,
    //                                                                          dim_m);

    // std::cout << "\t\tcal-grad - z=<x,w> completed."  << getCurrentTime() << std::endl;
    // // std::cout << "level of z=<x, w>: " << z_vec[1]->GetLevel() << std::endl;
    // checkAndUpdateMaxRAM("\t\tGradCalc / After z=<x,w>");
//     std::vector<Ciphertext<DCRTPoly>> y_pred_vec(dim_n); // Pre-size the vector
//     // --- 2. Sigmoid Activation: y_pred = sigmoid(z)
//     #pragma omp parallel for
//     for (size_t i = 0; i < dim_n; ++i) {
//         y_pred_vec[i] = cc->EvalPoly(z_vec[i], this->sigmoid_coeffs);
//     }
//     std::cout << "\t\tcal-grad - Sigmoid (EvalPoly) completed."  << getCurrentTime() << std::endl;
//     // std::cout << "level of sigmoid: " << y_pred_vec[1]->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tGradCalc / After Sigmoid");

//     // --- 3. CALCULATE GRADIENT CORE: gradients = (y_pred - Y) * X ---
//     std::vector<Ciphertext<DCRTPoly>> gradients_vec(dim_n);
//     // PARALLELIZED LOOP
//     #pragma omp parallel for
//     for (size_t i = 0; i < dim_n; ++i) {
//         Ciphertext<DCRTPoly> y_diff = cc->EvalAdd(y_pred_vec[i], cc->EvalMult(Y[i], -1));
//         Ciphertext<DCRTPoly> y_diff_extended = EncryptedValueToVec(cc, y_diff, dim_m);
//         gradients_vec[i] = cc->EvalMultAndRelinearize(y_diff_extended, input_X[i]);
//     }
//     std::cout << "\t\tcal-grad - Gradient core multiplication completed."  << getCurrentTime() << std::endl;
//     // std::cout << "level of (y_pred-y)*x: " << gradients_vec[1]->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tGradCalc / After Core Grad");

//     // 4. Average Gradient: grad_avg = sum(gradients) / n
//     std::vector<double> zeros(dim_m, 0.0);
//     Plaintext zero_plain = cc->MakeCKKSPackedPlaintext(zeros);
//     Ciphertext<DCRTPoly> sum = cc->Encrypt(publicKey, zero_plain);

//     for (size_t i = 0; i < dim_n; ++i) {
//         sum = cc->EvalAdd(sum, gradients_vec[i]);
//     }
//     checkAndUpdateMaxRAM("\t\tGradCalc / After Sum");

//     Ciphertext<DCRTPoly> grad_avg = cc->EvalMult(sum, (1.0 / dim_n));
//     std::cout << "\t\tcal-grad - Calculated gradient average." <<  getCurrentTime() << std::endl;
//     // std::cout << "level of grad_avg: " << grad_avg->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tGradCalc / After Avg");

//     // --- 5. Add Differential Privacy Noise
//     grad_avg = cc->EvalAdd(grad_avg, noise);
//     checkAndUpdateMaxRAM("\t\tGradCalc / After Noise Add (End)");
//     std::cout << "\t\tcal-grad - noise is added."  << getCurrentTime() << std::endl;

//     // ------------Path B: Barrier Calculation-----------------------
//     // --- 6. Calculate Log-Barrier Term (for weight constraint)
//     // --- 6.1 ||w||^2
//     Ciphertext<DCRTPoly> weight_norm = EncryptedSquaredNorm(cc, weights, dim_m, publicKey);
//     // std::cout << "level of ||w||^2: " << weight_norm->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tBarrier / After ||w||^2");
//     std::cout << "\t\tbarrier -  (||w||^2) is calculated."  << getCurrentTime() << std::endl;

//     // --- 6.2 (theta - ||w||^2)
//     Ciphertext<DCRTPoly> barrier_1 = cc->EvalAdd(this->theta, cc->EvalMult(weight_norm, -1));
//     // std::cout << "level of (theta - ||w||^2): " << barrier_1->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tBarrier / After (theta - ||w||^2)");
//     std::cout << "\t\tbarrier -  (theta - ||w||^2) is calculated."  << getCurrentTime() << std::endl;

//     // --- 6.3 barrier_2=1/(theta - ||w||^2)
//     // Ciphertext<DCRTPoly> barrier_2 = EncryptedInverseNR(cc, barrier_1, this->inverse_interval, 2, publicKey);
//     Ciphertext<DCRTPoly> barrier_2 = EncryptedInverseTS(cc, barrier_1, publicKey);
//     // std::cout << "level of inverse: " << barrier_2->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tBarrier / After inverse");
//     std::cout << "\t\tbarrier - inverseTS is calculated."  << getCurrentTime() << std::endl;

//     // --- 6.4 barrier = (2 * lambda * w) / (theta - ||w||^2)
//     double two_lambda = 2.0 * this->lambda;
//     Ciphertext<DCRTPoly> barrier_temp = cc->EvalMult(barrier_2, two_lambda);
//     Ciphertext<DCRTPoly> barrier = cc->EvalMultAndRelinearize(barrier_temp, weights);
//     std::cout << "\t\tcal-grad - Barrier term calculated."  << getCurrentTime() << std::endl;
//     // std::cout << "level of barrier: " << barrier->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tBarrier / After barrier term");
    
//     // ----------------------------------------------
//     // --- 7. Add Barrier Term to Gradient
//     grad_avg = cc->EvalAdd(grad_avg, barrier);
//     checkAndUpdateMaxRAM("\t\tGradCalc / After add barrier to grad");

//     return grad_avg;

 
// // ------------------------------------------------------------
// // parallel method 2
// // ------------------------------------------------------------
// ------------Path A: Gradient Calculation -----------------------
    // // --- 1. Linear Hypothesis: z = <x, w>
    // std::vector<Ciphertext<DCRTPoly>> z_vec = EncryptedMatrixVectorMultiply(cc,
    //                                                                          input_X,
    //                                                                          weights,
    //                                                                          dim_m);

    // std::cout << "\t\tcal-grad - z=<x,w> completed."  << getCurrentTime() << std::endl;
    // // std::cout << "level of z=<x, w>: " << z_vec[1]->GetLevel() << std::endl;
    // checkAndUpdateMaxRAM("\t\tGradCalc / After z=<x,w>");
//  // Storage
//     std::vector<Ciphertext<DCRTPoly>> gradients_vec(dim_n);
//     Ciphertext<DCRTPoly> barrier;

//     // ------------------------------------------------------------
//     // Parallel region with tasks
//     // ------------------------------------------------------------
//     #pragma omp parallel
//     {
//         #pragma omp single
//         {
//             // ====================================================
//             // Path A: Per-sample gradient (fully parallel, no barrier)
//             // ====================================================
//             std::cout << "\t\tcal-grad - Path A (gradient) started. "
//                       << getCurrentTime() << std::endl;
//             #pragma omp taskloop grainsize(1)
//             for (size_t i = 0; i < dim_n; ++i) {
//                 // Sigmoid
//                 Ciphertext<DCRTPoly> y_pred =
//                     cc->EvalPoly(z_vec[i], this->sigmoid_coeffs);

//                 // (y_pred - y)
//                 Ciphertext<DCRTPoly> y_diff =
//                     cc->EvalAdd(y_pred, cc->EvalMult(Y[i], -1));

//                 // Extend to vector
//                 Ciphertext<DCRTPoly> y_diff_extended =
//                     EncryptedValueToVec(cc, y_diff, dim_m);

//                 // Gradient contribution
//                 gradients_vec[i] =
//                     cc->EvalMultAndRelinearize(y_diff_extended, input_X[i]);
//             }

//             // ====================================================
//             // Path B: Barrier computation (independent task)
//             // ====================================================
    
//             #pragma omp task
//             {
//                 std::cout << "\t\tbarrier - Path B started. "
//                           << getCurrentTime() << std::endl;

//                 Ciphertext<DCRTPoly> weight_norm =
//                     EncryptedSquaredNorm(cc, weights, dim_m, publicKey);
//                 std::cout << "\t\tbarrier -  (||w||^2) is calculated."  << getCurrentTime() << std::endl;
//                 checkAndUpdateMaxRAM("\t\tBarrier / After ||w||^2");

//                 Ciphertext<DCRTPoly> barrier_1 =
//                     cc->EvalAdd(this->theta,
//                                 cc->EvalMult(weight_norm, -1));

//                 Ciphertext<DCRTPoly> barrier_2 =
//                     EncryptedInverseTS(cc, barrier_1, publicKey);

//                 Ciphertext<DCRTPoly> barrier_temp =
//                     cc->EvalMult(barrier_2, 2.0 * this->lambda);

//                 barrier =
//                     cc->EvalMultAndRelinearize(barrier_temp, weights);

//                 std::cout << "\t\tcal-grad - Barrier term calculated."
//                           << getCurrentTime() << std::endl;

//                 checkAndUpdateMaxRAM("\t\tBarrier / After barrier term");
//                 std::cout << "\t\tbarrier - Path B ended "
//                       << getCurrentTime() << std::endl;

//             }

//             // Wait for Path A and Path B to finish
//             #pragma omp taskwait
//             std::cout << "\t\tcal-grad - Path A (gradient) end. "
//                   << getCurrentTime() << std::endl;
//         }
//     }

//     // ------------------------------------------------------------
//     // 4. Average Gradient (serial reduction, correct semantics)
//     // ------------------------------------------------------------
//     std::vector<double> zeros(dim_m, 0.0);
//     Plaintext zero_plain = cc->MakeCKKSPackedPlaintext(zeros);
//     Ciphertext<DCRTPoly> sum = cc->Encrypt(publicKey, zero_plain);

//     for (size_t i = 0; i < dim_n; ++i) {
//         sum = cc->EvalAdd(sum, gradients_vec[i]);
//     }

//     checkAndUpdateMaxRAM("\t\tGradCalc / After Sum");

//     Ciphertext<DCRTPoly> grad_avg =
//         cc->EvalMult(sum, (1.0 / dim_n));

//     std::cout << "\t\tcal-grad - Calculated gradient average."
//               << getCurrentTime() << std::endl;

//     // ------------------------------------------------------------
//     // 5. Add DP noise + barrier
//     // ------------------------------------------------------------
//     grad_avg = cc->EvalAdd(grad_avg, noise);
//     grad_avg = cc->EvalAdd(grad_avg, barrier);

//     checkAndUpdateMaxRAM("\t\tGradCalc / End");

//     return grad_avg;


// // ------------------------------------------------------------
// // parallel method 3
// // ------------------------------------------------------------
// // ------------Path A: Gradient Calculation -----------------------
//     // --- 1. Linear Hypothesis: z = <x, w>
//     std::vector<Ciphertext<DCRTPoly>> z_vec = EncryptedMatrixVectorMultiply(cc,
//                                                                              input_X,
//                                                                              weights,
//                                                                              dim_m);

//     std::cout << "\t\tcal-grad - z=<x,w> completed."  << getCurrentTime() << std::endl;
//     // std::cout << "level of z=<x, w>: " << z_vec[1]->GetLevel() << std::endl;
//     checkAndUpdateMaxRAM("\t\tGradCalc / After z=<x,w>");
//  // Storage
//     std::vector<Ciphertext<DCRTPoly>> gradients_vec(dim_n);
//     Ciphertext<DCRTPoly> barrier;
//     std::cout << "\t\tcal-grad - Path A (gradient) started. "
//                     << getCurrentTime() << std::endl;
//     #pragma omp parallel for
//         // ====================================================
//         // Path A: Per-sample gradient (fully parallel, no barrier)
//         // ====================================================
    
//         for (size_t i = 0; i < dim_n+1; ++i) {
//             if (i==dim_n){
//             std::cout << "\t\tbarrier - Path B started. "
//                         << getCurrentTime() << std::endl;

//             Ciphertext<DCRTPoly> weight_norm =
//                 EncryptedSquaredNorm(cc, weights, dim_m, publicKey);
//             std::cout << "\t\tbarrier -  (||w||^2) is calculated."  << getCurrentTime() << std::endl;
//             checkAndUpdateMaxRAM("\t\tBarrier / After ||w||^2");

//             Ciphertext<DCRTPoly> barrier_1 =
//                 cc->EvalAdd(this->theta,
//                             cc->EvalMult(weight_norm, -1));

//             Ciphertext<DCRTPoly> barrier_2 =
//                 EncryptedInverseTS(cc, barrier_1, publicKey);

//             Ciphertext<DCRTPoly> barrier_temp =
//                 cc->EvalMult(barrier_2, 2.0 * this->lambda);

//             barrier =
//                 cc->EvalMultAndRelinearize(barrier_temp, weights);

//             std::cout << "\t\tcal-grad - Barrier term calculated."
//                         << getCurrentTime() << std::endl;

//             checkAndUpdateMaxRAM("\t\tBarrier / After barrier term");
//             std::cout << "\t\tbarrier - Path B ended "
//                     << getCurrentTime() << std::endl;
//         }

//             else{
//             // Sigmoid
//             Ciphertext<DCRTPoly> y_pred =
//                 cc->EvalPoly(z_vec[i], this->sigmoid_coeffs);

//             // (y_pred - y)
//             Ciphertext<DCRTPoly> y_diff =
//                 cc->EvalAdd(y_pred, cc->EvalMult(Y[i], -1));

//             // Extend to vector
//             Ciphertext<DCRTPoly> y_diff_extended =
//                 EncryptedValueToVec(cc, y_diff, dim_m);

//             // Gradient contribution
//             gradients_vec[i] =
//                 cc->EvalMultAndRelinearize(y_diff_extended, input_X[i]);
//         }
    
//     }

           
   
//     // ------------------------------------------------------------
//     // 4. Average Gradient (serial reduction, correct semantics)
//     // ------------------------------------------------------------
//     std::vector<double> zeros(dim_m, 0.0);
//     Plaintext zero_plain = cc->MakeCKKSPackedPlaintext(zeros);
//     Ciphertext<DCRTPoly> sum = cc->Encrypt(publicKey, zero_plain);

//     for (size_t i = 0; i < dim_n; ++i) {
//         sum = cc->EvalAdd(sum, gradients_vec[i]);
//     }

//     checkAndUpdateMaxRAM("\t\tGradCalc / After Sum");

//     Ciphertext<DCRTPoly> grad_avg =
//         cc->EvalMult(sum, (1.0 / dim_n));

//     std::cout << "\t\tcal-grad - Calculated gradient average."
//               << getCurrentTime() << std::endl;

//     // ------------------------------------------------------------
//     // 5. Add DP noise + barrier
//     // ------------------------------------------------------------
//     grad_avg = cc->EvalAdd(grad_avg, noise);
//     grad_avg = cc->EvalAdd(grad_avg, barrier);

//     checkAndUpdateMaxRAM("\t\tGradCalc / End");

//     return grad_avg;

// 
// // ------------------------------------------------------------
// // parallel method 4
// // ------------------------------------------------------------

//     std::vector<Ciphertext<DCRTPoly>> gradients_vec(dim_n);
//     Ciphertext<DCRTPoly> barrier;

//     std::cout << "\t\tcal-grad - Path A (gradient) started. "
//               << getCurrentTime() << std::endl;

//     // Parallel loop over all samples
//     #pragma omp parallel for
//     for (size_t i = 0; i < dim_n + 1; ++i) {
//         if (i == dim_n) {
//             // Barrier term (Path B) - computed only once
//             std::cout << "\t\tbarrier - Path B started. "
//                       << getCurrentTime() << std::endl;

//             Ciphertext<DCRTPoly> weight_norm =
//                 EncryptedSquaredNorm(cc, weights, dim_m, publicKey);
//             checkAndUpdateMaxRAM("\t\tBarrier / After ||w||^2");

//             Ciphertext<DCRTPoly> barrier_1 =
//                 cc->EvalAdd(this->theta, cc->EvalMult(weight_norm, -1));

//             Ciphertext<DCRTPoly> barrier_2 =
//                 EncryptedInverseTS(cc, barrier_1, publicKey);

//             Ciphertext<DCRTPoly> barrier_temp =
//                 cc->EvalMult(barrier_2, 2.0 * this->lambda);

//             barrier = cc->EvalMultAndRelinearize(barrier_temp, weights);

//             checkAndUpdateMaxRAM("\t\tBarrier / After barrier term");
//             std::cout << "\t\tbarrier - Path B ended "
//                       << getCurrentTime() << std::endl;
//         } else {
//             // Compute dot product z = <x_i, weights> inline
//             Ciphertext<DCRTPoly> mult = cc->EvalMultAndRelinearize(input_X[i], weights);

//             Ciphertext<DCRTPoly> sum = mult;
//             for (size_t j = 1; j < dim_m; j <<= 1) {
//                 auto rotated = cc->EvalAtIndex(sum, j);
//                 sum = cc->EvalAdd(sum, rotated);
//             }

//             // Apply sigmoid polynomial
//             Ciphertext<DCRTPoly> y_pred = cc->EvalPoly(sum, this->sigmoid_coeffs);

//             // Compute gradient contribution: (y_pred - y_i) * x_i
//             Ciphertext<DCRTPoly> y_diff = cc->EvalAdd(y_pred, cc->EvalMult(Y[i], -1));
//             Ciphertext<DCRTPoly> y_diff_extended = EncryptedValueToVec(cc, y_diff, dim_m);

//             gradients_vec[i] = cc->EvalMultAndRelinearize(y_diff_extended, input_X[i]);
//         }
//     }

//     // Average gradients
//     std::vector<double> zeros(dim_m, 0.0);
//     Plaintext zero_plain = cc->MakeCKKSPackedPlaintext(zeros);
//     Ciphertext<DCRTPoly> sum_grad = cc->Encrypt(publicKey, zero_plain);

//     for (size_t i = 0; i < dim_n; ++i) {
//         sum_grad = cc->EvalAdd(sum_grad, gradients_vec[i]);
//     }
//     checkAndUpdateMaxRAM("\t\tGradCalc / After Sum");

//     Ciphertext<DCRTPoly> grad_avg = cc->EvalMult(sum_grad, (1.0 / dim_n));
//     std::cout << "\t\tcal-grad - Calculated gradient average."
//               << getCurrentTime() << std::endl;

//     // Add DP noise + barrier
//     grad_avg = cc->EvalAdd(grad_avg, noise);
//     grad_avg = cc->EvalAdd(grad_avg, barrier);

//     checkAndUpdateMaxRAM("\t\tGradCalc / End");

//     return grad_avg;



// ------------------------------------------------------------
// parallel method 6
// ------------------------------------------------------------

    std::vector<Ciphertext<DCRTPoly>> gradients_vec(dim_n);
    Ciphertext<DCRTPoly> barrier;


    std::atomic<bool> pathA_recorded{false};
    double pathA_time = 0.0;

    double pathB_time = 0.0;

    // Parallel loop over all samples
    #pragma omp parallel
{
    #pragma omp single nowait
    {
        // Path B: exactly one task
        #pragma omp task
        {
            auto t0 = std::chrono::steady_clock::now();
            Ciphertext<DCRTPoly> weight_norm =
                EncryptedSquaredNorm(cc, weights, dim_m, publicKey);

            Ciphertext<DCRTPoly> barrier_1 =
                cc->EvalAdd(this->theta, cc->EvalMult(weight_norm, -1));

            // Ciphertext<DCRTPoly> barrier_2 =
            //     EncryptedInverseTS(cc, barrier_1, publicKey);
            // Ciphertext<DCRTPoly> barrier_2 =
            //     EncryptedInverseNR(cc, barrier_1, this->inverse_interval, 7, publicKey);

                Ciphertext<DCRTPoly> barrier_2 = cc->EvalPoly(barrier_1, this->p_kappa_coeffs);

            Ciphertext<DCRTPoly> barrier_temp =
                cc->EvalMult(barrier_2, 2.0 * this->lambda);

            barrier = cc->EvalMultAndRelinearize(barrier_temp, weights);

            auto t1 = std::chrono::steady_clock::now();
            pathB_time =std::chrono::duration<double>(t1 - t0).count();
            
        }

        // Path A: many tasks (or a taskloop)
        #pragma omp taskloop
        for (size_t i = 0; i < dim_n; ++i) {
            auto t0 = std::chrono::steady_clock::now();

            Ciphertext<DCRTPoly> mult =
                cc->EvalMultAndRelinearize(input_X[i], weights);

            Ciphertext<DCRTPoly> sum = mult;
            for (size_t j = 1; j < dim_m; j <<= 1)
                sum = cc->EvalAdd(sum, cc->EvalAtIndex(sum, j));

            Ciphertext<DCRTPoly> y_pred =
                cc->EvalPoly(sum, this->sigmoid_coeffs);

            Ciphertext<DCRTPoly> y_diff =
                cc->EvalAdd(y_pred, cc->EvalMult(Y[i], -1));

            Ciphertext<DCRTPoly> y_diff_ext =
                EncryptedValueToVec(cc, y_diff, dim_m);

            gradients_vec[i] =
                cc->EvalMultAndRelinearize(y_diff_ext, input_X[i]);

            auto t1  = std::chrono::steady_clock::now();
            double dt =
                    std::chrono::duration<double>(t1 - t0).count();
            if (!pathA_recorded.exchange(true,
                        std::memory_order_relaxed)) {
                    pathA_time = dt;
                }
        }
    }
}


    

    std::cout << std::fixed
            << "\t path-B : duration = " << pathB_time << " seconds\n";

    std::cout << std::fixed
            << "\t path-A : duration (one task) = "
            << pathA_time << " seconds\n";

    // Average gradients
    std::vector<double> zeros(dim_m, 0.0);
    Plaintext zero_plain = cc->MakeCKKSPackedPlaintext(zeros);
    Ciphertext<DCRTPoly> sum_grad = cc->Encrypt(publicKey, zero_plain);

    for (size_t i = 0; i < dim_n; ++i) {
        sum_grad = cc->EvalAdd(sum_grad, gradients_vec[i]);
    }
    // checkAndUpdateMaxRAM("\t\tGradCalc / After Sum");

    Ciphertext<DCRTPoly> grad_avg = cc->EvalMult(sum_grad, (1.0 / dim_n));
    std::cout << "\t\tcal-grad - Calculated gradient average."
              << getCurrentTime() << std::endl;

    // Add DP noise + barrier
    grad_avg = cc->EvalAdd(grad_avg, noise);
    grad_avg = cc->EvalAdd(grad_avg, barrier);

    // checkAndUpdateMaxRAM("\t\tGradCalc / End");

    return grad_avg;


};

Ciphertext<DCRTPoly> DP_GD_no_clipping::fit(
    const CryptoContext<DCRTPoly>& cc,
    size_t dim_n,
    size_t dim_m,
    const std::string& x_data_dir, 
    const std::string& y_data_dir, 
    const std::vector<Ciphertext<DCRTPoly>>& noise_list,
    uint32_t T,
    PublicKey<DCRTPoly>& publicKey,
    size_t batch_size,
    const std::string& weights_output_dir
) {

   //setup the folder for saving ct_weights
    const std::string checkpointDir =weights_output_dir;
    std::filesystem::create_directories(checkpointDir);

    // // Initialize weights to an encrypted vector of zeros
    // std::vector<double> zeros(dim_m, 0.0);
    // Plaintext zero_weights_plain = cc->MakeCKKSPackedPlaintext(zeros);
    // Ciphertext<DCRTPoly> weights = cc->Encrypt(publicKey, zero_weights_plain);
    // checkAndUpdateMaxRAM("\tFit / After Initial Weights Encrypt");

    const uint32_t resume_iter = 400;
    Ciphertext<DCRTPoly> weights;
    
    std::string weights_enc_path = checkpointDir + "/weights_noclip_paral"+"/weights_iter_"+std::to_string(resume_iter)+".bin";
    
    if (std::filesystem::exists(weights_enc_path)) {
        // Load weights as a single Ciphertext
        std::cout << "\n>>> RESUME MODE ACTIVE <<<" << std::endl;
        std::cout << "\tLoading weights from "<< weights_enc_path << std::endl;
        Serial::DeserializeFromFile(weights_enc_path, weights, SerType::BINARY);
        
        //boostrap if needed
        std::cout << "\tApplying immediate Bootstrap after loaded encryped weights..." << std::endl;
        weights = cc->EvalBootstrap(weights);
        std::cout << "\tWeights loaded and refreshed. Training will start from iteration " << resume_iter + 1 << "." << std::endl;
    } else {
        std::vector<double> zeros(dim_m, 0.0);
        Plaintext zero_weights_plain = cc->MakeCKKSPackedPlaintext(zeros);
        weights = cc->Encrypt(publicKey, zero_weights_plain);
        // std::cout << "\t[LEVEL] 'weights' (init) level after Encrypt: " << weights->GetLevel() << std::endl;
        // checkAndUpdateMaxRAM("\tFit / After Initial Weights Encrypt");
    }


    // //Setup for random mini-batch sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    // // fixed the random seed for testing only
    // std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> distrib(0, dim_n - 1);

    //loading the random index for all:
    // std::string batch_indices_path = checkpointDir + "/batch_index.csv";
    // std::vector<std::vector<size_t>> batch_indices=readCSVIndices(batch_indices_path);


    std::cout << "\t Start training for " << T << " iterations (mini-batch size = " 
              << batch_size << ")... " << getCurrentTime() << std::endl;

    // for (uint32_t t = 0; t < T; ++t) {
    uint32_t start_t = std::filesystem::exists(weights_enc_path) ? resume_iter : 0;
    for (uint32_t t = start_t; t < T; ++t) {
        
        auto iter_start = std::chrono::high_resolution_clock::now();
        std::cout << "\n\t Start iteration " << t + 1 << ": " << getCurrentTime()
                  << " memory: " << getCurrentMemoryMB() << " MB" << std::endl;
        // std::cout << "level at start of iteration: " << weights->GetLevel() << std::endl;
        // checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / Start");

        std::vector<Ciphertext<DCRTPoly>> X_batch(batch_size); 
        std::vector<Ciphertext<DCRTPoly>> Y_batch(batch_size); 

        ////option 1: generate random index
        std::vector<size_t> indices(batch_size);
        for(size_t i = 0; i < batch_size; ++i) {
            indices[i] = distrib(gen); // Select a random index *with replacement*
        }

        ////option 2: loading the random index for iteration t:
        // const std::vector<size_t>& indices = batch_indices[t];

        std::cout << "\t\tgenerate random mini-batch index (size " << batch_size << ")  ... ";
        // Load the batch from disk in parallel
        #pragma omp parallel for
        for(size_t i = 0; i < batch_size; ++i) {
            size_t rand_idx = indices[i]; 
            
            std::string x_file = x_data_dir + "x_" + std::to_string(rand_idx) + ".bin";
            std::string y_file = y_data_dir + "y_" + std::to_string(rand_idx) + ".bin";

            Serial::DeserializeFromFile(x_file, X_batch[i], SerType::BINARY);
            Serial::DeserializeFromFile(y_file, Y_batch[i], SerType::BINARY);
        }
        std::cout << "Done. " << getCurrentTime() << std::endl;

        // checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / Before grad_calc");
        Ciphertext<DCRTPoly> gradient = calculate_gradient(
            cc, batch_size, dim_m, X_batch, Y_batch, noise_list[t], weights, publicKey);
        // checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / After grad_calc");

        weights = cc->EvalAdd(weights, cc->EvalMult(gradient, -this->learning_rate));
        // checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / After Weight Update");
        
        // set the iteration for saving the weight
        if ((t + 1) >=T-100 || (t + 1)%50==0) {
           
            // weights_log.push_back(weights);

            // --- save ct_weights to local ---
            std::cout << "\t... Saving checkpoint for iteration " << (t + 1) << " ...\n";
            std::string weightsSaveDir = checkpointDir + "/weights_enc_noclip"; 
            std::filesystem::create_directories(weightsSaveDir);

            std::stringstream ss;
            ss << weightsSaveDir << "/weights_iter_" << (t + 1) << ".bin";
            std::string filename = ss.str();
            
            Serial::SerializeToFile(filename, weights, SerType::BINARY);
            std::cout << "\t... Checkpoint saved to " << filename << " ...\n";
     
        }

        // set the number of iteration to bootstrape
        if ((t + 1) % 3 == 0) {
            std::cout << "\n\t>>> BOOTSTRAPPING WEIGHTS AT ITERATION " << t + 1 << " <<<" << std::endl;
            auto start_boot = std::chrono::high_resolution_clock::now();
            std::cout << "Scale before bootstrap: " << weights->GetScalingFactor() << std::endl;
            std::cout << "level before bootstrap: " << weights->GetLevel() << std::endl;
            // checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / Before Bootstrap");
           
            weights = cc->EvalBootstrap(weights);
            std::cout << "Scale after bootstrap: " << weights->GetScalingFactor() << std::endl;
            std::cout << "level after bootstrap: " << weights->GetLevel() << std::endl;
            // checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / After Bootstrap");

            auto end_boot = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> boot_duration = end_boot - start_boot;
            std::cout << std::fixed 
                      << "\t>>> BOOTSTRAPPING COMPLETE. Duration: " 
                      << boot_duration.count() << " seconds. <<<\n" << std::endl;
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iter_duration = iter_end - iter_start;

        std::cout << std::fixed 
                  << "\t End iteration " << t + 1 << ": " << getCurrentTime()
                  << ", duration = " << iter_duration.count() << " seconds"
                  << ", memory: " << getCurrentMemoryMB() << " MB" << std::endl;
        // checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / End");
    };

    return weights;
}

// --- END OF DP_GD_no_clipping CLASS ---


int main(int argc, char* argv[]) {
    // checkAndUpdateMaxRAM("Program Start");
    std::cout << "\n====== SECURE MODEL TRAINING (LOADING FROM DISK) ========\n" << std::endl;
    std::string checkpointDir = "../" + std::string(argv[1]);
    std::string train_log = argv[2];
    std::cout << train_log << std::endl;
    std::string dataset_name = argv[5];
    
    #ifdef _OPENMP
        std::cout << "OpenMP is enabled. Max threads: " << omp_get_max_threads() << std::endl;
    #else
        std::cout << "OpenMP is NOT enabled. Running in serial." << std::endl;
    #endif

    // --- 1. Read Plaintext Data (ONLY to get dimensions) ---
    // This matches your original code's logic.
    auto start_read = std::chrono::high_resolution_clock::now();
    std::cout << "Start data reading (for dimensions)... " << getCurrentTime() << std::endl;

    std::string x_path = "../preprocess_datasets/" + dataset_name + "_x_train.csv";
    std::vector<std::vector<double>> X_data = readCSV(x_path);

    // std::vector<std::vector<double>> X_data = readCSV("../preprocess_datasets/adult_x_train.csv");
    auto end_read = std::chrono::high_resolution_clock::now();
    std::cout << "Finished data reading in " << std::chrono::duration<double>(end_read - start_read).count() << " seconds." << std::endl;


    // --- 2. Training Parameters ---
    double learning_rate = std::stod(argv[6]);    //0.041
    int T                = std::stoi(argv[7]);   //1000
    double theta         = std::stod(argv[8]);
    double lambda        = std::stod(argv[9]);
    size_t batch_size    = std::stoi(argv[10]);

    size_t dim_n = X_data.size(); 
    // size_t dim_n = 5; //for test
    size_t dim_m_org = X_data.empty() ? 0 : X_data[0].size();
    size_t dim_m = dim_m_org+1;
    
    X_data.clear(); // Clear memory, we don't need plaintext X anymore

    std::vector<int32_t> rotationIndices;
    for (int i = 1; i < dim_m; i++) {
        rotationIndices.push_back(i);
    }

    std::cout << "Data size: " << dim_n << " x " << dim_m << ", Training Iterations: " << T << std::endl;


    // --- 3. Load CryptoContext and Keys ---
    // std::string checkpointDir = "../cipher_1";
    std::cout << "Checkpoint directory: " << checkpointDir << std::endl;

    std::string cc_path = checkpointDir + "/cryptocontext.bin";
    std::string pk_path = checkpointDir + "/key_public.bin";
    std::string sk_path = checkpointDir + "/key_secret.bin";

    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;

    std::cout << "Loading existing CryptoContext and Keys..." << getCurrentTime() << std::endl;
    
    if (!std::filesystem::exists(cc_path) || !std::filesystem::exists(pk_path) || !std::filesystem::exists(sk_path)) {
        std::cerr << "Error: CryptoContext or Key files not found." << std::endl;
        std::cerr << "Please run the 'setup_encrypt' program first." << std::endl;
        return 1;
    }

    Serial::DeserializeFromFile(cc_path, cc, SerType::BINARY);
    Serial::DeserializeFromFile(pk_path, keyPair.publicKey, SerType::BINARY);
    Serial::DeserializeFromFile(sk_path, keyPair.secretKey, SerType::BINARY);
    
    std::cout << "CryptoContext and Keys loaded successfully." << std::endl;

    std::cout << "Regenerating evaluation keys from loaded secret key_1..." << std::endl;

    // std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> levelBudget;
    try {
        levelBudget.push_back(std::stoul(argv[3]));
        levelBudget.push_back(std::stoul(argv[4]));
    }
    catch (const std::exception& e) {
        // This will catch errors if argv[3] or [4] is not a number
        std::cerr << "Error: Invalid levelBudget argument. Must be a number." << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    cc->EvalBootstrapSetup(levelBudget);

    // 1. Regenerate Multiplication Keys (Fixes your error)
    cc->EvalMultKeyGen(keyPair.secretKey);

    // 2. Regenerate Rotation Keys (Needed for EncryptedMatrixVectorMultiply)
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
    
    // 3. Regenerate Bootstrap Keys (Needed for EvalBootstrap)
    uint32_t numSlots = cc->GetEncodingParams()->GetBatchSize();
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);


    // --- 4. Load Encrypted Data ---
    auto start_enc = std::chrono::high_resolution_clock::now();

    // Define paths to the *directories* you created in setup_encrypt.cpp
    std::string x_enc_dir = checkpointDir + "/X_enc_individual/";
    std::string y_enc_dir = checkpointDir + "/Y_enc_individual/";
    std::string noise_enc_path = checkpointDir + "/noise_enc.bin"; 

    // We only load the noise vector now. X and Y will be loaded on-demand.
    std::vector<Ciphertext<DCRTPoly>> noise_enc; 

    if (!std::filesystem::exists(noise_enc_path)) {
        std::cout << "Encrypted noise file not found at: " << noise_enc_path << std::endl;
        std::cout << "Attempting to generate from CSV..." << getCurrentTime() << std::endl;
        
        // 1. Define path and read plaintext noise CSV
        std::string noise_path = "../noise_noclip_new/" + dataset_name + "_noise.csv";
        if (!std::filesystem::exists(noise_path)) {
             std::cerr << "Error: Plaintext noise file not found at " << noise_path << std::endl;
             std::cerr << "Cannot proceed without noise data. Please create the CSV or the encrypted .bin file." << std::endl;
             return 1;
        }
        
        std::cout << "\tReading plaintext noise from " << noise_path << "..." << std::endl;
        std::vector<std::vector<double>> noise_list_data = readCSV(noise_path);
        std::cout << "\tRead " << noise_list_data.size() << " noise vectors." << std::endl;

        // 2. Encrypt the noise
        std::cout << "\tEncrypting noise vectors (in parallel)..." << std::endl;
        noise_enc.resize(noise_list_data.size()); // Pre-size the vector for parallel insertion
        
        #pragma omp parallel for
        for (size_t i = 0; i < noise_list_data.size(); ++i) {
            std::vector<double> noise_vec = noise_list_data[i];
            // Ensure noise vector matches feature dimension (pad with 0s if needed)
            noise_vec.resize(dim_m, 0.0); 
            
            Plaintext pt_noise = cc->MakeCKKSPackedPlaintext(noise_vec);
            noise_enc[i] = cc->Encrypt(keyPair.publicKey, pt_noise);
        }
        std::cout << "\t...Encryption complete." << std::endl;

        // 3. Save the encrypted noise for future runs
        std::cout << "\tSaving newly encrypted noise to " << noise_enc_path << "..." << std::endl;
        Serial::SerializeToFile(noise_enc_path, noise_enc, SerType::BINARY);
        std::cout << "\t... Encrypted noise saved for next run." << std::endl;
    }
    else {
        // Original logic: file exists, so just load it.
        std::cout << "Loading encrypted noise from disk: " << noise_enc_path << "..." << getCurrentTime() << std::endl;
        Serial::DeserializeFromFile(noise_enc_path, noise_enc, SerType::BINARY); 
        std::cout << "Encrypted noise loaded successfully." << std::endl;
    }
    noise_enc.resize(T);
        
    // std::cout << "Finished loading all encrypted data." << std::endl;
    // std::cout << "Finished loading noise data." << std::endl;
    
    auto end_enc = std::chrono::high_resolution_clock::now();
    std::cout << "Finished loading noise data. " << std::chrono::duration<double>(end_enc - start_enc).count() << " seconds. Memory: " << getCurrentMemoryMB() << " MB" << std::endl;

    // --- 5. Model Initialization and Training ---
    std::cout << "Initialize model..." << getCurrentTime() << std::endl;
    DP_GD_no_clipping model(learning_rate, T, 1.0, 0.5, theta, lambda);

    auto start_train = std::chrono::high_resolution_clock::now();
    std::cout << "Start training..." << getCurrentTime() << std::endl;
    // Ciphertext<DCRTPoly> ct_weights = model.fit(
    //     cc, dim_n, dim_m, X_E_enc, Y_E_enc, noise_enc, T, keyPair.publicKey, batch_size, checkpointDir
    // );

    Ciphertext<DCRTPoly> ct_weights = model.fit(
        cc, dim_n, dim_m, x_enc_dir, y_enc_dir, noise_enc, T, keyPair.publicKey, batch_size, checkpointDir
    );
    
    auto end_train = std::chrono::high_resolution_clock::now();
    std::cout << "End training in " << std::chrono::duration<double>(end_train - start_train).count() << " seconds. Memory: " << getCurrentMemoryMB() << " MB" << std::endl;
    // checkAndUpdateMaxRAM("After Training");
    // --- 6. Decrypt and Display Results ---
    PrintWeights(cc, keyPair.secretKey, ct_weights, dim_m, T);
    std::cout << "\n====== FINAL MAX RAM USAGE: " << max_ram_mb << " MB ======\n" << std::endl;
    return 0;
}