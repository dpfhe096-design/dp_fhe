#include "openfhe.h"

using namespace lbcrypto;


CryptoContext<DCRTPoly> ccEnvSetup(int m, int n);


// Given the cipher text of one value x
// return a encrypted vector of length
// each item of which is x
Ciphertext<DCRTPoly> EncryptedValueToVec(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& x,
    size_t l
);


Ciphertext<DCRTPoly> PackEncryptedVector(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ciphertexts,
    const PublicKey<DCRTPoly>& publicKey);

    
std::vector<Ciphertext<DCRTPoly>> UnpackEncryptedVector(
    CryptoContext<DCRTPoly> cc,
    Ciphertext<DCRTPoly> packedCiphertext,
    size_t length,
    PublicKey<DCRTPoly>& publicKey);


/**
 * @brief Compares two ciphertexts c1 and c2 as (c1 > c2).
 * 
 * This function compares two ciphertexts using the Chebyshev approximation of
 * the sign function. The result is a ciphertext, where a value of 1 indicates
 * that the corresponding components of c1 are greater than those of c2, a
 * value of 0 indicates that the corresponding components of c1 are less than
 * those of c2, and a value of 0.5 indicates that the corresponding components
 * of c1 are approximately equal to those of c2.
 * 
 * @param c1 The first ciphertext to be compared.
 * @param c2 The second ciphertext to be compared.
 * @param a The lower bound of the approximation interval.
 * @param b The upper bound of the approximation interval.
 * @param degree The degree of the Chebyshev polynomial approximation.
 * @param error The threshold for considering values close to zero (default is
 * 0.00001).
 * @return Ciphertext<DCRTPoly> A ciphertext representing the output of the
 * comparison (c1 > c2).
 */

Ciphertext<DCRTPoly> compare(
    const Ciphertext<DCRTPoly> &c1,
    const Ciphertext<DCRTPoly> &c2,
    const double a,
    const double b,
    const uint32_t degree,
    const double error
);


// Input two cipher text c1 and c2,
// return the max of the two
Ciphertext<DCRTPoly> max(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly> &c1,
    const Ciphertext<DCRTPoly> &c2,
    const double a,
    const double b,
    const uint32_t degree
);


std::vector<Ciphertext<DCRTPoly>> EncryptedMatrixVectorMultiply(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& encMatrixRows, // each row is a ciphertext
    const Ciphertext<DCRTPoly>& encVec, // encrypted vector
    size_t dim  // dimension of the vector
);




// Sum up an encrypted vector of length len
// return the ciphertext of the sum
Ciphertext<DCRTPoly> EncryptedVecSumAll(
    CryptoContext<DCRTPoly> cc,
    Ciphertext<DCRTPoly>& encVec,
    size_t len);


// Multiply a matrix of size rows*cols with a vector of length c 
std::vector<Ciphertext<DCRTPoly>> EncryptedMatrixVectorMultiply(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& encMatrix,
    // std::vector<std::vector<Ciphertext<DCRTPoly>>>& encMatrix,
    const Ciphertext<DCRTPoly>& encVector,
    size_t dim);


// Calculate the L2 norm of an encrypted vector
Ciphertext<DCRTPoly> EncryptedL2Norm(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& encVector,
    size_t len,
    std::vector<double> coeffs,
PublicKey<DCRTPoly>& publicKey);


Ciphertext<DCRTPoly> EncryptedL2Norm(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& encVector,
    size_t len,
    std::vector<double> coeffs,
    PublicKey<DCRTPoly>& publicKey);


// Calculate the sigmoid function on the encrypted value
// Using Chebyshev approximation
Ciphertext<DCRTPoly> EncryptedSigmoid(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> z,
    std::vector<double> coeffs,
    PublicKey<DCRTPoly>& publicKey);


// // Overloaded version with vector z as input
// Ciphertext<DCRTPoly> EncryptedSigmoid(
//     const CryptoContext<DCRTPoly>& cc,
//     std::vector<Ciphertext<DCRTPoly>> z,
//     KeyPair<DCRTPoly> keyPair);



// Calculate the inverse function (1/x) on the encrypted value
// Using Chebyshev approximation
Ciphertext<DCRTPoly> EncryptedInverse(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    std::vector<double> coeffs,
    PublicKey<DCRTPoly>& publicKey);



Ciphertext<DCRTPoly> EncryptedSquaredNorm(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& encVector,
    size_t len, 
    PublicKey<DCRTPoly>& publicKey);


Ciphertext<DCRTPoly> EncryptedInverseNR(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    std::vector<double> interval,   // interval = {a, b}
    int iter,                       // number of Newton iterations
    PublicKey<DCRTPoly>& publicKey);

Ciphertext<DCRTPoly> EncryptedInverseTS(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    PublicKey<DCRTPoly>& publicKey);