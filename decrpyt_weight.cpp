#include "openfhe.h"
#include "cryptocontext-ser.h"
#include "ciphertext-ser.h"
#include "key/key-ser.h"
#include "utils/serial.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <stdexcept> // For std::invalid_argument, std::out_of_range

using namespace lbcrypto;
namespace fs = std::filesystem;

// --- Decrypt and Display full weight at iteration iter ---
void PrintWeights(
    const CryptoContext<DCRTPoly>& cc, 
    const PrivateKey<DCRTPoly>& sk,
    const Ciphertext<DCRTPoly>& ct,
    size_t length) { 

    Plaintext pl_weights;
    cc->Decrypt(sk, ct, &pl_weights);
    pl_weights->SetLength(length);

    auto weights_vec = pl_weights->GetCKKSPackedValue();

    std::cout << "("; // Added size info for debugging
    std::cout << std::fixed << std::setprecision(8); // Use fixed precision

    for (size_t j = 0; j < weights_vec.size(); ++j) {
        std::cout << weights_vec[j].real();
        if (j < weights_vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./decrypt_weights <checkpointDir> <vector_length>\n";
        return 1;
    }

    std::string checkpointDir = "../" + std::string(argv[1]);
    size_t vec_length = 0;
    try {
         vec_length = std::stoul(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid vector_length argument: " << argv[2] << std::endl;
        return 1;
    }


    std::string cc_path = checkpointDir + "/cryptocontext.bin";
    std::string sk_path = checkpointDir + "/key_secret.bin";

    std::cout << "Loading CryptoContext and Secret Key..." << std::endl;

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(cc_path, cc, SerType::BINARY)) {
        std::cerr << "Error: Failed to load cryptocontext from " << cc_path << std::endl;
        return 1;
    }

    PrivateKey<DCRTPoly> sk;
    if (!Serial::DeserializeFromFile(sk_path, sk, SerType::BINARY)) {
        std::cerr << "Error: Failed to load secret key from " << sk_path << std::endl;
        return 1;
    }

    std::cout << "Loaded CryptoContext and Secret Key successfully.\n" << std::endl;

    // --- Step 1: Collect all matching file paths ---
    std::string weightsDir = checkpointDir + "/"+"weights_enc_1";
    std::vector<fs::path> weight_files;

    // Check if weightsDir exists
    if (!fs::exists(weightsDir) || !fs::is_directory(weightsDir)) {
        std::cerr << "Error: Weights directory not found: " << weightsDir << std::endl;
        return 1;
    }

    const std::string file_prefix = "weights_iter_";
    const std::string file_suffix = ".bin";

    for (const auto& entry : fs::directory_iterator(weightsDir)) {
        if (!fs::is_regular_file(entry)) continue;

        std::string filename = entry.path().filename().string();
        
        // Use starts_with and ends_with for clearer logic
        if (filename.rfind(file_prefix, 0) == 0 && 
            filename.length() > (file_prefix.length() + file_suffix.length()) && // Must have at least one char for the number
            filename.rfind(file_suffix) == (filename.length() - file_suffix.length())) 
        {
            weight_files.push_back(entry.path());
        } else if (filename.rfind(file_prefix, 0) == 0) {
            std::cout << "Skipping file (does not match pattern correctly): " << filename << std::endl;
        }
    }

    if (weight_files.empty()) {
        std::cerr << "No valid weight files found in " << weightsDir << std::endl;
        return 0; // Not an error, just no files to process
    }

    // --- Step 2: Sort the vector numerically ---
    std::sort(weight_files.begin(), weight_files.end(), 
        [&](const fs::path& a, const fs::path& b) {
        
        std::string f_a = a.filename().string();
        std::string f_b = b.filename().string();

        int num_a = 0;
        int num_b = 0;

        try {
            // More robust parsing: find string between prefix and suffix
            size_t start_a = file_prefix.length();
            size_t end_a = f_a.rfind(file_suffix);
            if (end_a > start_a) {
                num_a = std::stoi(f_a.substr(start_a, end_a - start_a));
            } else {
                 std::cerr << "Warning: Could not parse number from " << f_a << ", treating as 0." << std::endl;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: std::invalid_argument while parsing " << f_a << ", treating as 0." << std::endl;
        } catch (const std::out_of_range& e) {
             std::cerr << "Warning: std::out_of_range while parsing " << f_a << ", treating as 0." << std::endl;
        }

        try {
            size_t start_b = file_prefix.length();
            size_t end_b = f_b.rfind(file_suffix);
            if (end_b > start_b) {
                num_b = std::stoi(f_b.substr(start_b, end_b - start_b));
            } else {
                std::cerr << "Warning: Could not parse number from " << f_b << ", treating as 0." << std::endl;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: std::invalid_argument while parsing " << f_b << ", treating as 0." << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: std::out_of_range while parsing " << f_b << ", treating as 0." << std::endl;
        }

        return num_a < num_b;
    });

    // --- Step 3: Process the files in sorted order ---
    std::cout << "\n--- Processing " << weight_files.size() << " file(s) ---\n" << std::endl;
    for (const auto& path : weight_files) {
        std::string filename = path.filename().string();
        std::cout << "Decrypting " << filename << "..." << std::endl;

        Ciphertext<DCRTPoly> ct;
        if (!Serial::DeserializeFromFile(path.string(), ct, SerType::BINARY)) {
            std::cerr << "Error: Failed to load " << filename << std::endl;
            continue;
        }

        PrintWeights(cc, sk, ct, vec_length);
    }
    // std::cout << "\nDecryption complete." << std::endl;
    return 0;
}
