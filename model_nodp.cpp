#define PROFILE // No-DP training with regularization 

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

// --- Helper for debugging CKKs precision/range issues ---
Plaintext TestDecryption(
    const CryptoContext<DCRTPoly>& cc,
    const PrivateKey<DCRTPoly>& sk,
    const Ciphertext<DCRTPoly>& ct,
    size_t length = 0) {

    Plaintext pl;
    cc->Decrypt(sk, ct, &pl);

    
    // 2. Set the length (number of slots) if a specific length is requested
    size_t actual_length = (length == 0) ? ct->GetSlots() : length;
    pl->SetLength(actual_length);

    return pl;
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

// --- PASTE YOUR DP_GD_clipping CLASS HERE ---
// (The class definition and its member functions: calculate_gradient and fit)

// code for training DP-GD with clipping
class DP_GD_clipping {
public:
    // Parameters:
    double learning_rate;
    int T;
    double C;
    double loss;


    // // Noise and Coeffs:
    // std::vector<std::vector<double>> noise_list;
    // std::vector<double> sigmoid_coeffs;
    // std::vector<double> l2norm_coeffs;
    // std::vector<double> inverse_coeffs;
    // std::vector<double> inverseNR_interval;
    // int inverseNR_iteration;
    
    // ==== TEST: Only for testing ====
    // KeyPair<DCRTPoly> keyPair;


    // Logs
    std::vector<Ciphertext<DCRTPoly>> weights_log;
    std::vector<Ciphertext<DCRTPoly>> loss_log;


    // Model parameters initialization
    DP_GD_clipping(
        double learning_rate,
        int T,
        double C,
        double loss) : learning_rate(learning_rate),
                         T(T),
                         C(C),
                         loss(loss){};
    // Coefficients for polynomial approximations
    std::vector<double> sigmoid_coeffs = {
        // sigmoid coeff for compas
        // 0.5, 0.1697807, 0.0, -0.0031215, 0.0, 0.0000270, 0.0, -0.0000001
        // sigmoid coeff for mnist
        0.5, 0.15266035, 0, -0.00221240, 0, 0.00001614, 0, -0.00000005
        // sigmoid coeff for adult
        // 0.5, 0.1778298, 0.0, -0.0036922, 0.0, 0.0000366, 0.0, -0.0000001
       // sigmoid coeff for credit
        // 0.5, 0.19282043, 0, -0.00514345, 0, 0.00007417, 0, -0.00000048
    };

    std::vector<double> l2norm_coeffs={
    0.6253335362143935, 0.3959562477220996, -0.01983042717565514, 0.0007276954759648647, -1.5600617699380835e-05, 1.8934212216839758e-07, -1.2051355117954955e-09, 3.122032538422496e-12    
    };
    

    // Merged and optimized function to compute the entire gradient in one pass.
    Ciphertext<DCRTPoly> calculate_gradient(
        const CryptoContext<DCRTPoly>& cc,
        size_t dim_n,
        size_t dim_m,
        const std::vector<Ciphertext<DCRTPoly>>& input_X,
        const std::vector<Ciphertext<DCRTPoly>>& Y,
        const Ciphertext<DCRTPoly>& noise,
        const Ciphertext<DCRTPoly>& weights,
        PublicKey<DCRTPoly>& publicKey,
        PrivateKey<DCRTPoly>& secretKey);

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
        PrivateKey<DCRTPoly>& secretKey,
        size_t batch_size,
        const std::string& weights_output_dir
    );
};


Ciphertext<DCRTPoly> DP_GD_clipping::calculate_gradient(
    const CryptoContext<DCRTPoly>& cc,
    size_t dim_n,
    size_t dim_m,
    const std::vector<Ciphertext<DCRTPoly>>& input_X,
    const std::vector<Ciphertext<DCRTPoly>>& Y,
    const Ciphertext<DCRTPoly>& noise,
    const Ciphertext<DCRTPoly>& weights,
    PublicKey<DCRTPoly>& publicKey,
    PrivateKey<DCRTPoly>& secretKey) {
    checkAndUpdateMaxRAM("\t\tGradCalc / Start");

    // --- 1. Linear Hypothesis: z = <x, w>
    // Note: Levels of X and Y are assumed to be high (fresh from disk)
    // std::cout << "\t\t[LEVEL] 'weights' level at start of grad: " << weights->GetLevel() << std::endl;
    std::vector<Ciphertext<DCRTPoly>> z_vec = EncryptedMatrixVectorMultiply(cc, input_X, weights, dim_m);
    std::cout << "\t\tcal-grad - z=<x, w> completed."  << getCurrentTime() << std::endl;
    // if (!z_vec.empty()) {
    //     std::cout << "\t\t[LEVEL] 'z_vec[0]' level after MatVecMult: " << z_vec[0]->GetLevel() << std::endl;
    // }
    checkAndUpdateMaxRAM("\t\tGradCalc / After z=<x,w>");
    // // --- DEBUG: Check z_vec range ---
    // double z_max = -std::numeric_limits<double>::infinity();
    // double z_min = std::numeric_limits<double>::infinity();
    // for(size_t i = 0; i < dim_n; ++i) { 
    //     Plaintext z_plain = TestDecryption(cc, secretKey, z_vec[i], 1); 
    //     double z_value = z_plain->GetCKKSPackedValue()[0].real(); 
        
    //     // 3. Compare and update overall max/min
    //     if (z_value > z_max) {
    //         z_max = z_value;
    //     }
    //     if (z_value < z_min) {
    //         z_min = z_value;
    //     }
    // }
    // std::cout << "the max z value across all samples is " << std::fixed << std::setprecision(5) 
    //           << z_max << "; the min z value is " << z_min << ";" << std::endl;
    // std::cout << std::defaultfloat; // Reset precision formatting


    // --- 2. Sigmoid Activation: y_pred = sigmoid(z)
    std::vector<Ciphertext<DCRTPoly>> y_pred_vec(dim_n); // Pre-size the vector
    #pragma omp parallel for
    for (size_t i = 0; i < dim_n; ++i) {
        y_pred_vec[i] = cc->EvalPoly(z_vec[i], this->sigmoid_coeffs);
    }
    std::cout << "\t\tcal-grad - Sigmoid (EvalPoly) completed."  << getCurrentTime() << std::endl;
    // if (!y_pred_vec.empty()) {
    //     std::cout << "\t\t[LEVEL] 'y_pred_vec[0]' level after EvalPoly: " << y_pred_vec[0]->GetLevel() << std::endl;
    // }
    checkAndUpdateMaxRAM("\t\tGradCalc / After Sigmoid");

    // // --- DEBUG: Check y_pred_vec range ---    
    // // Initialize doubles to hold the overall max/min across all samples
    // double y_max = -std::numeric_limits<double>::infinity();
    // double y_min = std::numeric_limits<double>::infinity();

    // for(size_t i = 0; i < dim_n; ++i) { 
    //     // 1. Decrypt one sample, specifying length=1 to check the first slot
    //     // We assume the first slot holds the single predicted value for this sample.
    //     Plaintext y_plain = TestDecryption(cc, secretKey, y_pred_vec[i], 1); 
        
    //     // 2. Extract the actual double value from the first slot (index 0)
    //     double y_value = y_plain->GetCKKSPackedValue()[0].real(); 
        
    //     // 3. Compare and update overall max/min
    //     if (y_value > y_max) {
    //         y_max = y_value;
    //     }
    //     if (y_value < y_min) {
    //         y_min = y_value;
    //     }
    // }
    
    // // 4. Output the results
    // std::cout << "the max y_pred value across all samples is " << std::fixed << std::setprecision(5) 
    //           << y_max << "; the min y_pred value is " << y_min << ";" << std::endl;
    // std::cout << std::defaultfloat; // Reset precision formatting
  


    // --- 3. CALCULATE GRADIENT CORE: gradients = (y_pred - Y) * X ---
    std::vector<Ciphertext<DCRTPoly>> gradients_vec(dim_n);
    // PARALLELIZED LOOP
    #pragma omp parallel for
    for (size_t i = 0; i < dim_n; ++i) {
        Ciphertext<DCRTPoly> y_diff_mult = cc->EvalMult(Y[i], -1);
        Ciphertext<DCRTPoly> y_diff = cc->EvalAdd(y_pred_vec[i], y_diff_mult);
        // Note: Can't print level inside parallel loop easily, will print one after
        
        Ciphertext<DCRTPoly> y_diff_extended = EncryptedValueToVec(cc, y_diff, dim_m);
        gradients_vec[i] = cc->EvalMultAndRelinearize(y_diff_extended, input_X[i]);
    }
    std::cout << "\t\tcal-grad - Gradient core multiplication completed."  << getCurrentTime() << std::endl;
    // if (!gradients_vec.empty()) {
    // std::cout << "\t\t[LEVEL] 'gradients_vec[0]' level after core mult: " << gradients_vec[0]->GetLevel() << std::endl;
    // }
    checkAndUpdateMaxRAM("\t\tGradCalc / After Core Grad");


    // ==========================================================================
    // __________________________________________________________________________
    // ____________DO SGD WITHOUT DP, comment out all clipping steps_____________
    // ==========================================================================

    // // --- 4. Calculate clipped gradient ---
    // // --- 4.1 ||gradient||
    // std::vector<Ciphertext<DCRTPoly>> gradient_norm_vec(dim_n);
    // // This loop is serial, so we can print levels
    // #pragma omp parallel for
    // for (size_t i = 0; i < dim_n; i++){
    //     // std::cout << "\t cal-grad - L2 norm... " <<  getCurrentTime() << " line " << i << std::endl;
    //     gradient_norm_vec[i] = EncryptedL2Norm(cc, gradients_vec[i], dim_m, l2norm_coeffs, publicKey);
    //     // std::cout << "\t\t[LEVEL] 'gradient_norm_vec[" << i << "]' level after L2Norm: " << gradient_norm_vec[i]->GetLevel() << std::endl;
    //     // gradient_norm_vec[i] = cc->EvalBootstrap(gradient_norm_vec[i]);
    // }
    // std::cout << "\t\t clip-grad - L2 norm of gradient is completed."  << getCurrentTime() << std::endl;
    // // std::cout << "\t\t[LEVEL] 'gradient_norm_vec level after L2Norm: " << gradient_norm_vec[0]->GetLevel() << std::endl;
    // checkAndUpdateMaxRAM("\t\tGradCalc / After L2Norm loop");
    // // // --- DEBUG: Check gradient_norm_vec range ---
    // // // Initialize doubles to hold the overall max/min L2 norm across all samples
    // // double norm_max = -std::numeric_limits<double>::infinity();
    // // double norm_min = std::numeric_limits<double>::infinity();

    // // for(size_t i = 0; i < dim_n; ++i) { 
    // //     Plaintext norm_plain = TestDecryption(cc, secretKey, gradient_norm_vec[i], 1); 
        
    // //     double norm_value = norm_plain->GetCKKSPackedValue()[0].real(); 
     
    // //     if (norm_value > norm_max) {
    // //         norm_max = norm_value;
    // //     }
    // //     if (norm_value < norm_min) {
    // //         norm_min = norm_value;
    // //     }
    // // }
    
    // // std::cout << "the max L2 norm across all samples is " << std::fixed << std::setprecision(5) 
    // //           << norm_max << "; the min L2 norm is " << norm_min << ";" << std::endl;
    // // std::cout << std::defaultfloat; // Reset precision formatting


    
    // // --- 4.2 gradient / max(1, ||gradient||)
    // std::vector<double> one (1, 1);
    // Ciphertext<DCRTPoly> cipher_one = cc->Encrypt(publicKey, cc->MakeCKKSPackedPlaintext(one));
    // // std::cout << "\t\t[LEVEL] 'cipher_one' level after Encrypt: " << cipher_one->GetLevel() << std::endl;
    

    // // The value to be multiplied: 1 / max(1, gradient_norm)
    // std::vector<Ciphertext<DCRTPoly>> clipped_gradient_vec(dim_n);
    // checkAndUpdateMaxRAM("\t\tGradCalc / Before Clipping loop");

    // #pragma omp parallel for
    // for (size_t i = 0; i < dim_n; i++){
        
    //     bool isFirstItem = (i == 0); 
    //     if (isFirstItem) checkAndUpdateMaxRAM("\t\tClipLoop[0] / Start");

    //     // std::cout << "\t cal-grad - compare norm with 1... " <<  getCurrentTime() << " line " << i << std::endl;
    //     // Ciphertext<DCRTPoly> mult_val = max(cc, cipher_one, gradient_norm_vec[i], 0, 100, 9);
    //     Ciphertext<DCRTPoly> mult_val = max(cc, cipher_one, gradient_norm_vec[i], 0, 100, 9);
    //     // std::cout << "\t\t[LEVEL] level after compare max{1, ||grad||} : " << mult_val->GetLevel() << std::endl;
    //     if (isFirstItem) checkAndUpdateMaxRAM("\t\tClipLoop[0] / After max()");
        
    //     //TEST: Only for testing
    //     // TestDecryption(cc, mult_val, "mult_val dec - 1: ", keyPair);
        
    //     // mult_val = cc->EvalBootstrap(mult_val);

    //     // std::cout << "3 Current NumberCiphertextElements = " << mult_val->NumberCiphertextElements() << std::endl;


    //     // extend encryption of one mult_val to a whole vector
    //     // std::cout << "\t cal-grad - Extend max to vec... " <<  getCurrentTime() << " line " << i << std::endl;
    //     mult_val = EncryptedValueToVec(cc, mult_val, dim_m);
    //     // std::cout << "\t\t[LEVEL] level re-format max valuetovector (before bootstrap)  : " << mult_val->GetLevel() << std::endl;
    //     if (isFirstItem) checkAndUpdateMaxRAM("\t\tClipLoop[0] / After ValueToVec");
        
    //     // TestDecryption(cc, mult_val, "mult_val dec - 4: ", keyPair);
        
    //     //// mult_val = 1 / mult_val
    //     // mult_val = cc->EvalBootstrap(mult_val);
    //     // std::cout << "\t\t[LEVEL] level re-format max valuetovector (after bootstrap) : " << mult_val->GetLevel() << std::endl;
    //     // std::cout << "\t\t[LEVEL] level before inverse : " << mult_val->GetLevel() << std::endl;
    //     // std::cout << "\t cal-grad - Inverse... " <<  getCurrentTime() << " line " << i << std::endl;
    //     // mult_val = EncryptedInverseNR(cc, mult_val, inverseNR_interval, inverseNR_iteration, publicKey);
    //     mult_val = EncryptedInverseTS(cc, mult_val, publicKey);
    //     // std::cout << "\t\t[LEVEL] level after inverse 1/||max|| : " << mult_val->GetLevel() << std::endl;
    //     if (isFirstItem) checkAndUpdateMaxRAM("\t\tClipLoop[0] / After Inverse()");

    //     clipped_gradient_vec[i] = (cc->EvalMultAndRelinearize(gradients_vec[i], mult_val));
    //     if (isFirstItem) checkAndUpdateMaxRAM("\t\tClipLoop[0] / After Final Mult");
    // }
    // std::cout << "\t\t clip-grad - grad/(max{1, l2-norm}) is completed"  << getCurrentTime() << std::endl;
    // // Print level for the first element after the parallel loop
    // // if (!clipped_gradient_vec.empty()) {
    //     // std::cout << "\t\t[LEVEL] 'clipped_gradient_vec[0]' level after final mult: " << clipped_gradient_vec[0]->GetLevel() << std::endl;
    // // }
    // checkAndUpdateMaxRAM("\t\tGradCalc / After Clipping loop");


    // ==========================================================================
    // __________________________________________________________________________
    // ____________DO SGD WITHOUT DP, comment out all clipping steps_____________
    // ==========================================================================



    // --- 5. Average Gradient: grad_avg = sum(gradients) / n ---
    std::vector<double> zeros(dim_m, 0.0);
    Plaintext zero_plain = cc->MakeCKKSPackedPlaintext(zeros);
    Ciphertext<DCRTPoly> sum = cc->Encrypt(publicKey, zero_plain);
    
    // for (size_t i = 0; i < dim_n; ++i) {
    //     sum = cc->EvalAdd(sum, clipped_gradient_vec[i]);
    // }

     for (size_t i = 0; i < dim_n; ++i) {
        sum = cc->EvalAdd(sum, gradients_vec[i]);
    }


    // std::cout << "\t\t[LEVEL] 'sum' level after all additions: " << sum->GetLevel() << std::endl;
    checkAndUpdateMaxRAM("\t\tGradCalc / After Sum");

    Ciphertext<DCRTPoly> grad_avg = cc->EvalMult(sum, (1.0 / dim_n));
    std::cout << "\t\tcal-grad - Calculated gradient average." <<  getCurrentTime() << std::endl;
    // std::cout << "\t\t[LEVEL] 'grad_avg' level after mult by 1/n: " << grad_avg->GetLevel() << std::endl;
    checkAndUpdateMaxRAM("\t\tGradCalc / After Avg");

    // // --- 6. Add Differential Privacy Noise
    // // std::cout << "\t\t[LEVEL] 'noise' level: " << noise->GetLevel() << std::endl;
    // grad_avg = cc->EvalAdd(grad_avg, noise);
    // // std::cout << "\t\t[LEVEL] 'grad_avg' level after adding noise: " << grad_avg->GetLevel() << std::endl;
    // checkAndUpdateMaxRAM("\t\tGradCalc / After Noise Add (End)");

    
    //TEST: Only for testing
    // TestDecryption(cc, grad_avg, "grad_avg dec: ", keyPair);

    return grad_avg;
};




Ciphertext<DCRTPoly> DP_GD_clipping::fit(
    const CryptoContext<DCRTPoly>& cc,
    size_t dim_n,
    size_t dim_m,
    const std::string& x_data_dir, 
    const std::string& y_data_dir, 
    const std::vector<Ciphertext<DCRTPoly>>& noise_list,
    uint32_t T,
    PublicKey<DCRTPoly>& publicKey,
    PrivateKey<DCRTPoly>& secretKey,
    size_t batch_size,
    const std::string& weights_output_dir
) {

   //setup the folder for saving ct_weights
    const std::string checkpointDir =weights_output_dir;
    std::filesystem::create_directories(checkpointDir);

    //// Initialize weights to an encrypted vector of zeros
    // std::vector<double> zeros(dim_m, 0.0);
    // Plaintext zero_weights_plain = cc->MakeCKKSPackedPlaintext(zeros);
    // Ciphertext<DCRTPoly> weights = cc->Encrypt(publicKey, zero_weights_plain);
    // // std::cout << "\t[LEVEL] 'weights' (init) level after Encrypt: " << weights->GetLevel() << std::endl;
    // checkAndUpdateMaxRAM("\tFit / After Initial Weights Encrypt");


    // in case the training is not fully complete, restart the train by loading saved weights_enc
    const uint32_t resume_iter = 1000;
    Ciphertext<DCRTPoly> weights;
    
    std::string weights_enc_path = checkpointDir + "/weights_enc_nodp"+"/weights_iter_"+std::to_string(resume_iter)+".bin";
    
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
        checkAndUpdateMaxRAM("\tFit / After Initial Weights Encrypt");
    }


    //Setup for random mini-batch sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    // fixed the random seed for testing only
    // std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> distrib(0, dim_n - 1);


    std::cout << "\t Start training for " << T << " iterations (mini-batch size = " 
              << batch_size << ")... " << getCurrentTime() << std::endl;

    // for (uint32_t t = 0; t < T; ++t) {
    uint32_t start_t = std::filesystem::exists(weights_enc_path) ? resume_iter : 0;
    for (uint32_t t = start_t; t < T; ++t) {
        
        auto iter_start = std::chrono::high_resolution_clock::now();
        std::cout << "\n\t Start iteration " << t + 1 << ": " << getCurrentTime()
                  << " memory: " << getCurrentMemoryMB() << " MB" << std::endl;

        checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / Start");
        
        // std::cout << "\t[LEVEL] 'weights' level at start of iter " << (t+1) << ": " << weights->GetLevel() << std::endl;
        // std::cout << "\t[LEVEL] 'noise_list[" << t << "]' level: " << noise_list[t]->GetLevel() << std::endl;


        // --- Create Mini-Batch (Sampling with Replacement) ---
        // ... (omitted for brevity, no crypto ops) ...
        std::vector<Ciphertext<DCRTPoly>> X_batch(batch_size); // Pre-size
        std::vector<Ciphertext<DCRTPoly>> Y_batch(batch_size); // Pre-size
        std::vector<size_t> indices(batch_size);
        for(size_t i = 0; i < batch_size; ++i) {
            indices[i] = distrib(gen); // Select a random index *with replacement*
        }

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
        // Assuming X/Y ciphertexts are at max level, no need to print

        checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / Before grad_calc");


        Ciphertext<DCRTPoly> gradient = calculate_gradient(
            cc, batch_size, dim_m, X_batch, Y_batch, noise_list[t], weights, publicKey, secretKey);
        
        // std::cout << "\t[LEVEL] 'gradient' level after calculate_gradient: " << gradient->GetLevel() << std::endl;


        Ciphertext<DCRTPoly> scaled_gradient = cc->EvalMult(gradient, -this->learning_rate);
        // std::cout << "\t[LEVEL] 'scaled_gradient' level after mult by LR: " << scaled_gradient->GetLevel() << std::endl;
        checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / After grad_calc");


        weights = cc->EvalAdd(weights, scaled_gradient);
        //// --- debug: Decrypt the weights for every iteration: ---
        // Plaintext weights_plain = TestDecryption(cc, secretKey, weights, dim_m); // Decrypt the full vector
        // auto weights_vec = weights_plain->GetCKKSPackedValue();
        // if (!weights_vec.empty()) {
        //     // Print the first value (slot 0) and the last value to confirm successful decryption
        //     std::cout << "\t[CHECK] Decryption of weights succeeded: (";
        //     for (size_t j = 0; j < weights_vec.size(); ++j) {
        //         std::cout << weights_vec[j].real();
        //         if (j < weights_vec.size() - 1) {
        //             std::cout << ", ";
        //         }
        //     }
        //     std::cout << ")" << std::endl;
        //     std::cout << std::defaultfloat; // Reset precision formatting
        // } else {
        //     std::cout << "\t[ALERT] Decryption of weights failed or resulted in an empty vector." << std::endl;
        // }

        // std::cout << "\t[LEVEL] 'weights' level after gradient update: " << weights->GetLevel() << std::endl;
        checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / After Weight Update");

        // set the iteration for saving the weight
        if ((t + 1) >=T-100 || (t + 1)%50==0) {
           
            // weights_log.push_back(weights);

            // --- save ct_weights to local ---
            std::cout << "\t... Saving checkpoint for iteration " << (t + 1) << " ...\n";
            std::string weightsSaveDir = checkpointDir + "/weights_enc_nodp"; 
            std::filesystem::create_directories(weightsSaveDir);

            std::stringstream ss;
            ss << weightsSaveDir << "/weights_iter_" << (t + 1) << ".bin";
            std::string filename = ss.str();
            
            Serial::SerializeToFile(filename, weights, SerType::BINARY);
            std::cout << "\t... Checkpoint saved to " << filename << " ...\n";
     
        }

        
           
        // set the number of iteration to boostrape
        if ((t + 1) % 3 == 0) {
            std::cout << "\n\t>>> BOOTSTRAPPING WEIGHTS AT ITERATION " << t + 1 << " <<<" << std::endl;
            auto start_boot = std::chrono::high_resolution_clock::now();
            std::cout << "Scale before bootstrap: " << weights->GetScalingFactor() << std::endl;
            // This is the level tracking you asked for
            std::cout << "level before bootstrap: " << weights->GetLevel() << std::endl; 
            checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / Before Bootstrap");
           
            weights = cc->EvalBootstrap(weights);

            std::cout << "Scale after bootstrap: " << weights->GetScalingFactor() << std::endl;
            // This is the level tracking you asked for
            std::cout << "level after bootstrap: " << weights->GetLevel() << std::endl;
            checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / After Bootstrap");

            auto end_boot = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> boot_duration = end_boot - start_boot;
            std::cout << std::fixed
                      << "\t>>> BOOTSTRAPPING COMPLETE. Duration: " << boot_duration.count() 
                      << " seconds. <<<\n" << std::endl;
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iter_duration = iter_end - iter_start;

        std::cout << std::fixed 
                  << "\t End iteration " << t + 1 << ": " << getCurrentTime()
                  << ", duration = " << iter_duration.count() << " seconds"
                  << ", memory: " << getCurrentMemoryMB() << " MB" << std::endl;
        checkAndUpdateMaxRAM("\tIter " + std::to_string(t+1) + " / End");
    };

    return weights;
}

// --- END OF DP_GD_clipping CLASS ---


int main(int argc, char* argv[]) {
    checkAndUpdateMaxRAM("Program Start");
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
    size_t batch_size    = std::stoi(argv[8]);

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


    // Define paths to the *directories* you created in File 1
    std::string x_enc_dir = checkpointDir + "/X_enc_individual/";
    std::string y_enc_dir = checkpointDir + "/Y_enc_individual/";
    std::string noise_enc_path = checkpointDir + "/noise_clip_enc.bin"; 

    // We only load the noise vector now. X and Y will be loaded on-demand.
    std::vector<Ciphertext<DCRTPoly>> noise_enc; 

    if (!std::filesystem::exists(noise_enc_path)) {
        std::cout << "Encrypted noise file not found at: " << noise_enc_path << std::endl;
        std::cout << "Attempting to generate from CSV..." << getCurrentTime() << std::endl;
        
        // 1. Define path and read plaintext noise CSV
        std::string noise_path = "../noise_clip/" + dataset_name + "_noise.csv";
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
    
    // Check if noise_enc has enough data *before* resizing
    if (noise_enc.size() < T) {
         std::cerr << "CRITICAL ERROR: Not enough noise vectors for T iterations." << std::endl;
         std::cerr << "    Found: " << noise_enc.size() << " noise vectors" << std::endl;
         std::cerr << "    Need:  " << T << " noise vectors" << std::endl;
         std::cerr << "    Please generate a larger noise CSV file." << std::endl;
         return 1; // Exit
    }
    // Now it's safe to truncate
    noise_enc.resize(T);
    // std::cout << "\t[LEVEL] 'noise_enc[0]' level after load: " << noise_enc[0]->GetLevel() << std::endl;

        
    // std::cout << "Finished loading all encrypted data." << std::endl;
    // std::cout << "Finished loading noise data." << std::endl;
    
    auto end_enc = std::chrono::high_resolution_clock::now();
    std::cout << "Finished loading noise data. " << std::chrono::duration<double>(end_enc - start_enc).count() << " seconds. Memory: " << getCurrentMemoryMB() << " MB" << std::endl;

    // --- 5. Model Initialization and Training ---
    // ... (omitted, no crypto ops) ...


    std::cout << "Initialize model..." << getCurrentTime() << std::endl;
    DP_GD_clipping model(learning_rate, T, 1.0, 0.5);

    auto start_train = std::chrono::high_resolution_clock::now();
    std::cout << "Start training..." << getCurrentTime() << std::endl;
    // Ciphertext<DCRTPoly> ct_weights = model.fit(
    //     cc, dim_n, dim_m, X_E_enc, Y_E_enc, noise_enc, T, keyPair.publicKey, batch_size, checkpointDir
    // );

    Ciphertext<DCRTPoly> ct_weights = model.fit(
        cc, dim_n, dim_m, x_enc_dir, y_enc_dir, noise_enc, T, keyPair.publicKey, keyPair.secretKey, batch_size, checkpointDir
    );
    
    auto end_train = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << "End training in " << std::chrono::duration<double>(end_train - start_train).count() << " seconds. Memory: " << getCurrentMemoryMB() << " MB" << std::endl;
    checkAndUpdateMaxRAM("After Training");
    // --- 6. Decrypt and Display Results ---
    PrintWeights(cc, keyPair.secretKey, ct_weights, dim_m, T);
    std::cout << "\n====== FINAL MAX RAM USAGE: " << max_ram_mb << " MB ======\n" << std::endl;
    return 0;
}
