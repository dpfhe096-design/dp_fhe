#define PROFILE // turns on the reporting of timing results

#include "openfhe.h"
#include "util/crypto.h"  // Assuming this contains getCurrentTime
#include "util/data_prep.h" // Assuming this contains readCSV
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

using namespace lbcrypto;
// Current resident set size in MB
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

using namespace lbcrypto;


int main(int argc, char* argv[]) {
    std::cout << "\n====== DATA ENCRYPTION & KEY GENERATION ========\n" << std::endl;

    std::string checkpointDir = "../" + std::string(argv[1]);
    std::string train_log = argv[2];
    std::cout << train_log << std::endl;
    std::string dataset_name = argv[5];


    // --- 1. Data Preparation (Plaintext) ---
    auto start_read = std::chrono::high_resolution_clock::now();
    std::cout << "Start data reading... " << getCurrentTime() << std::endl;

    // Build the file paths
    std::string x_path = "../preprocess_datasets/" + dataset_name + "_x_train.csv";
    std::string y_path = "../preprocess_datasets/" + dataset_name + "_y_train.csv";
    std::string noise_path = "../noise_noclip/" + dataset_name + "_noise.csv";

    // Read the files directly
    std::vector<std::vector<double>> X_data = readCSV(x_path);
    std::vector<std::vector<double>> Y_data = readCSV(y_path);
    std::vector<std::vector<double>> noise_list_data = readCSV(noise_path);

    auto end_read = std::chrono::high_resolution_clock::now();
    std::cout << "Finished data reading in " << std::chrono::duration<double>(end_read - start_read).count() << " seconds." << std::endl;

    if (X_data.empty() || Y_data.empty() || noise_list_data.empty()) {
        std::cerr << "Error: Failed to read input data files. Exiting." << std::endl;
        return 1;
    }

    for(size_t i = 0; i < X_data.size(); ++i) { // Modify all loaded data with bias
            X_data[i].insert(X_data[i].begin(), 1.0);
    }

    // --- 2. Parameters (from your test setup) ---
    int T = 1000; // Used to determine how many noise vectors to encrypt
    size_t dim_n = X_data.size(); // Use this for full data
    // size_t dim_n = 5; // For test
    size_t dim_m = X_data.empty() ? 0 : X_data[0].size();
    
    // Resize data to match test size
    X_data.resize(dim_n);
    Y_data.resize(dim_n);
    noise_list_data.resize(T);

    std::cout << "Data size (for encryption): " << dim_n << " x " << dim_m 
              << ", Noise vectors: " << T << std::endl;

    // --- 3. Key Generation for Rotations ---
    std::vector<int32_t> rotationIndices;
    for (int i = 1; i < dim_m; i++) {
        rotationIndices.push_back(i);
    }

    // --- 4. Define Checkpoint Paths ---
    // std::string checkpointDir = "../enc_mnist";
    std::filesystem::create_directory(checkpointDir); // Ensure dir exists
    std::cout << "Checkpoint directory: " << checkpointDir << std::endl;

    std::string cc_path = checkpointDir + "/cryptocontext.bin";
    std::string pk_path = checkpointDir + "/key_public.bin";
    std::string sk_path = checkpointDir + "/key_secret.bin";
    // std::string x_enc_path = checkpointDir + "/X_enc.bin";     //// save encryption into one file
    // std::string y_enc_path = checkpointDir + "/Y_enc.bin";      //// save encryption into one file
    std::string noise_enc_path = checkpointDir + "/noise_enc.bin";

    // --- 5. Generate CryptoContext and Keys ---
    std::cout << "Generating new CryptoContext and Keys..." << getCurrentTime() << std::endl;
    
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    // parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(32768);
    // parameters.SetRingDim(65536);
    parameters.SetKeySwitchTechnique(HYBRID);
    // parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
    usint dcrtBits = 59;
    usint firstMod = 60;
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    
    uint32_t levelsAvailableAfterBootstrap = std::stoi(argv[6]);
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


    usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY);
    parameters.SetMultiplicativeDepth(depth);
    
    auto bd = FHECKKSRNS::GetBootstrapDepth(levelBudget, UNIFORM_TERNARY);
    std::cout << "Bootstrap depth (internal): " << bd << std::endl;
    std::cout << "Multiplicative depth total: " << depth << std::endl;

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    KeyPair<DCRTPoly> keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    
    uint32_t numSlots = cc->GetEncodingParams()->GetBatchSize();
    cc->EvalBootstrapSetup(levelBudget);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
    
    cc->EvalRotateKeyGen(keyPair.secretKey, rotationIndices);
    
    std::cout << "Bootstrap setup and key generation finished. With depth after boot = " 
    << levelsAvailableAfterBootstrap << std::endl;
            // Print Crypto Settings ---
    std::cout << "Crypto Settings: SetRingDim= " << parameters.GetRingDim() 
                << ", dcrtBits=" << dcrtBits <<", firstMod=" <<firstMod
                <<", levelBudget=" <<levelBudget
                << ", LevelsAvailable=" << levelsAvailableAfterBootstrap
                <<", SetScalingTechnique(FLEXIBLEAUTOEXT);"
                << std::endl;

    // --- 6. SAVE CONTEXT AND KEYS ---
    std::cout << "Saving new CryptoContext and Keys..." << std::endl;
    if (!Serial::SerializeToFile(cc_path, cc, SerType::BINARY)) {
        std::cerr << "Error serializing CryptoContext" << std::endl;
        return 1;
    }
    if (!Serial::SerializeToFile(pk_path, keyPair.publicKey, SerType::BINARY)) {
        std::cerr << "Error serializing public key" << std::endl;
        return 1;
    }
    if (!Serial::SerializeToFile(sk_path, keyPair.secretKey, SerType::BINARY)) {
        std::cerr << "Error serializing secret key" << std::endl;
        return 1;
    }
    std::cout << "Context and keys saved." << std::endl;

    // --- 7. Encrypt and Save All Data ---
    auto start_enc = std::chrono::high_resolution_clock::now();
    std::cout << "Encrypting X, Y, and Noise..." << getCurrentTime() << std::endl;

    // double initScale = pow(2.0, 16);  
    // std::cout << "with initial scale = " << initScale << std::endl;

////-----------encrypt in serial-------------------
    // std::vector<Ciphertext<DCRTPoly>> X_enc;
    // X_enc.reserve(dim_n);
    // for (size_t i = 0; i < dim_n; i++) {
    //     Plaintext p = cc->MakeCKKSPackedPlaintext(X_data[i]);
    //     // Plaintext p = cc->MakeCKKSPackedPlaintext(X_data[i], initScale);
    //     X_enc.push_back(cc->Encrypt(keyPair.publicKey, p));
    // }

    // std::vector<Ciphertext<DCRTPoly>> Y_enc;
    // Y_enc.reserve(dim_n);
    // for (size_t i = 0; i < dim_n; i++) {
    //     Plaintext p = cc->MakeCKKSPackedPlaintext(Y_data[i]);
    //     // Plaintext p = cc->MakeCKKSPackedPlaintext(Y_data[i], initScale);
    //     Y_enc.push_back(cc->Encrypt(keyPair.publicKey, p));
    // }
    
    // std::vector<Ciphertext<DCRTPoly>> noise_enc;
    // noise_enc.reserve(T);
    // for (size_t i = 0; i < T; i++) {
    //     Plaintext p = cc->MakeCKKSPackedPlaintext(noise_list_data[i]);
    //     // Plaintext p = cc->MakeCKKSPackedPlaintext(noise_list_data[i], initScale);
    //     noise_enc.push_back(cc->Encrypt(keyPair.publicKey, p));
    // }
////-----------encrypt in serial above-------------------



////-----------encrypt in parallel-------------------
    int nthreads = omp_get_max_threads();
    std::cout << "Using " << nthreads << " threads for encryption." << std::endl;

    // Create subdirectories for individual ciphertexts
    std::string x_dir = checkpointDir + "/X_enc_individual/";
    std::string y_dir = checkpointDir + "/Y_enc_individual/";
    std::filesystem::create_directory(x_dir);
    std::filesystem::create_directory(y_dir);
    std::cout << "Saving individual X/Y ciphertexts to: " << x_dir << " and " << y_dir << std::endl;

    // std::vector<Ciphertext<DCRTPoly>> X_enc(dim_n);   ////save encryption into one file
    // std::vector<Ciphertext<DCRTPoly>> Y_enc(dim_n);   ////save encryption into one file
    std::vector<Ciphertext<DCRTPoly>> noise_enc(T);

    #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < dim_n; i++) {
            Plaintext p = cc->MakeCKKSPackedPlaintext(X_data[i]);
            // X_enc[i] = cc->Encrypt(keyPair.publicKey, p);
            Ciphertext<DCRTPoly> ct = cc->Encrypt(keyPair.publicKey, p);
            // Save this single ciphertext to its own file
            std::string ct_path = x_dir + "x_" + std::to_string(i) + ".bin";
            Serial::SerializeToFile(ct_path, ct, SerType::BINARY);
        }
    std::cout<<"done encrypt x with Memory: " << getCurrentMemoryMB() << " MB, " << getCurrentTime() << std::endl;

    #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < dim_n; i++) {
            Plaintext p = cc->MakeCKKSPackedPlaintext(Y_data[i]);
            // Y_enc[i] = cc->Encrypt(keyPair.publicKey, p);
            Ciphertext<DCRTPoly> ct = cc->Encrypt(keyPair.publicKey, p);

            // Save this single ciphertext to its own file
            std::string ct_path = y_dir + "y_" + std::to_string(i) + ".bin";
            Serial::SerializeToFile(ct_path, ct, SerType::BINARY);
        }
    std::cout<<"done encrypt y with Memory: " << getCurrentMemoryMB() << " MB, " << getCurrentTime() << std::endl;

    #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < T; i++) {
            Plaintext p = cc->MakeCKKSPackedPlaintext(noise_list_data[i]);
            noise_enc[i] = cc->Encrypt(keyPair.publicKey, p);
        }
    std::cout<<"done encrypt noise with Memory: " << getCurrentMemoryMB() << " MB, " << getCurrentTime() << std::endl;
////-----------encrypt in parallel above-------------------

    std::cout << "Encryption complete. Saving to disk..." << std::endl;

    // Serial::SerializeToFile(x_enc_path, X_enc, SerType::BINARY);    ////save encryption into one file
    // Serial::SerializeToFile(y_enc_path, Y_enc, SerType::BINARY);    ////save encryption into one file
    Serial::SerializeToFile(noise_enc_path, noise_enc, SerType::BINARY); 
    
    std::cout << "Encrypted data saved to " << checkpointDir << std::endl;
    
    auto end_enc = std::chrono::high_resolution_clock::now();
    std::cout << "End data preparation and encryption in " << std::chrono::duration<double>(end_enc - start_enc).count() << " seconds. Memory: " << getCurrentMemoryMB() << " MB" << std::endl;

    std::cout << "\n====== ENCRYPTION COMPLETE ========\n" << std::endl;
    return 0;
}
