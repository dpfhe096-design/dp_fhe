
#include "openfhe.h"
#include "crypto.h"
#include "format.h"
#include <omp.h> 

using namespace lbcrypto;


CryptoContext<DCRTPoly> ccEnvSetup(int m, int n){
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(60);
    // parameters.SetBatchSize(8);
    parameters.SetSecurityLevel(HEStd_NotSet);
    // parameters.SetSecurityLevel(HEStd_128_classic); //Standard
    parameters.SetRingDim(2048);
    // parameters.SetRingDim(16384);
    
    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    usint dcrtBits               = 59;
    usint firstMod               = 60;

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    // Add rotation keys needed for EvalAtIndex (used in EvalSum)
    std::vector<int32_t> rotation_indices;
    for (int i = 1; i < m; i <<= 1)
        rotation_indices.push_back(i);
    // TODO

    return cc;
}

//////origional one
// // Given the cipher text of one value x
// // return a encrypted vector of length
// // each item of which is x
// Ciphertext<DCRTPoly> EncryptedValueToVec(
//     CryptoContext<DCRTPoly> cc,
//     const Ciphertext<DCRTPoly>& x,
//     size_t l
// )
// {
//     Ciphertext<DCRTPoly> result = x;

//     for (usint i = 1; i < l; i <<= 1) {
//         // Rotate and add
//         auto rotated = cc->EvalAtIndex(result, -i);
//         result = cc->EvalAdd(result, rotated);
//     }

//     return result;
// }

///// updated solve the approximation error bug
Ciphertext<DCRTPoly> EncryptedValueToVec(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& x, // Assumes value is in slot 0
    size_t l
)
{
    // 1. Create a mask: [1.0, 0.0, 0.0, ...]
    std::vector<double> maskVec(cc->GetEncodingParams()->GetBatchSize(), 0.0);
    maskVec[0] = 1.0;
    auto mask = cc->MakeCKKSPackedPlaintext(maskVec);

    // 2. Isolate the value in slot 0.
    // This gives [v, 0, 0, 0, ...]
    Ciphertext<DCRTPoly> result = cc->EvalMult(x, mask);
    
    // 3. !! IMPORTANT: Rescale after multiplying with plaintext !!
    // cc->ModReduceInPlace(result); 

    // 4. Now perform the broadcast.
    // I've changed -i to +i for a right-shift, which is
    // more standard, but your -i logic also works.
    for (usint i = 1; i < l; i <<= 1) {
        // Rotate and add
        auto rotated = cc->EvalAtIndex(result, -i); // Right-shift
        result = cc->EvalAdd(result, rotated);
    }

    return result;
}



// Pack vector of ciphertexts into one ciphertext
Ciphertext<DCRTPoly> PackEncryptedVector(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& ciphertexts,
    PublicKey<DCRTPoly>& publicKey) {

    size_t n = ciphertexts.size();
    std::vector<Ciphertext<DCRTPoly>> rotatedCiphertexts(n);

    for (size_t i = 0; i < n; ++i) {
        if (i == 0) {
            rotatedCiphertexts[i] = ciphertexts[i]; // No rotation needed
        } else {
            // Rotate ciphertext i by i to put value into slot i
            rotatedCiphertexts[i] = cc->EvalAtIndex(ciphertexts[i], -i);
        }
    }

    // Sum all rotated ciphertexts
    Ciphertext<DCRTPoly> packed = rotatedCiphertexts[0];
    for (size_t i = 1; i < n; ++i) {
        packed = cc->EvalAdd(packed, rotatedCiphertexts[i]);
    }

    return packed;
}



// Unpack one ciphertext of a vector into a vector of ciphertexts, each encrypting one value
std::vector<Ciphertext<DCRTPoly>> UnpackEncryptedVector(
    CryptoContext<DCRTPoly> cc,
    Ciphertext<DCRTPoly> packedCiphertext,
    size_t length,
    PublicKey<DCRTPoly>& publicKey) 
{
    std::vector<Ciphertext<DCRTPoly>> unpacked(length);

    for (size_t i = 0; i < length; ++i) {
        // Rotate slot i to position 0 (use negative index)
        auto rotated = cc->EvalAtIndex(packedCiphertext, static_cast<int>(i));

        // Mask to keep only slot 0
        std::vector<double> maskVec(cc->GetEncodingParams()->GetBatchSize(), 0.0);
        maskVec[0] = 1.0;
        auto mask = cc->MakeCKKSPackedPlaintext(maskVec);

        unpacked[i] = cc->EvalMult(rotated, mask);
        // cc->ModReduceInPlace(unpacked[i]); // <<< ADD ModReduce for safety with PT mult
        if(!unpacked[i]) throw std::runtime_error("EvalMult failed in UnpackEncryptedVector");
    }
    
    
    return unpacked;
}



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
)
{
    return c1->GetCryptoContext()->EvalChebyshevFunction(
        [error](double x) -> double { 
            if      (x > error)   return 1;
            else if (x >= -error) return 0.5;
            else                  return 0;
        },
        c1 - c2,
        a, b, degree
    );
}


// Input two cipher text c1 and c2,
// return the max of the two
Ciphertext<DCRTPoly> max(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly> &c1,
    const Ciphertext<DCRTPoly> &c2,
    const double a,
    const double b,
    const uint32_t degree
)
{
    
    auto comparison = compare(c1, c2, a, b, degree, 0.00001);
    auto diff = cc->EvalAdd(c1, cc->EvalMult(c2, -1));
    auto result = cc->EvalMultAndRelinearize(comparison, diff);
    result = cc->EvalAdd(result, c2);

    return result;
}


// using OpenMP to do parallel computation
std::vector<Ciphertext<DCRTPoly>> EncryptedMatrixVectorMultiply(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& encMatrixRows, // each row is a ciphertext
    const Ciphertext<DCRTPoly>& encVec, // encrypted vector
    size_t dim  // dimension of the vector
) {
    // UPDATED: Pre-size the vector for thread-safe assignments
    std::vector<Ciphertext<DCRTPoly>> resultRows(encMatrixRows.size());

    // UPDATED: This pragma tells OpenMP to split the work of this for-loop
    // across multiple threads. Each thread will handle a different chunk of rows.

    #pragma omp parallel for
    for (size_t i = 0; i < encMatrixRows.size(); ++i) {
        // Element-wise multiplication of row i and encVec
        // std::cout << "\t MatVecMul - EvalMul... " <<  getCurrentTime() << " line " << i << std::endl;
        Ciphertext<DCRTPoly> mult = cc->EvalMultAndRelinearize(encMatrixRows[i], encVec);

        // cc->ModReduceInPlace(mult);

        // Sum all elements in the vector to simulate dot product
        // std::cout << "\t MatVecMul - Sum vec up... " <<  getCurrentTime() << " line " << i << std::endl;
        Ciphertext<DCRTPoly> sum = mult;
        for (size_t j = 1; j < dim; j <<= 1) {
            auto rotated = cc->EvalAtIndex(sum, j);
            sum = cc->EvalAdd(sum, rotated);
        }

        // Assign the result to its correct spot. This is thread-safe.
        resultRows[i] = sum;
    }

    return resultRows;
}


// // origional code
// std::vector<Ciphertext<DCRTPoly>> EncryptedMatrixVectorMultiply(
//     CryptoContext<DCRTPoly> cc,
//     const std::vector<Ciphertext<DCRTPoly>>& encMatrixRows, // each row is a ciphertext
//     const Ciphertext<DCRTPoly>& encVec, // encrypted vector
//     size_t dim  // dimension of the vector
// ) {
//     std::vector<Ciphertext<DCRTPoly>> resultRows;

//     for (size_t i = 0; i < encMatrixRows.size(); ++i) {
        

//         // Element-wise multiplication of row i and encVec
//         std::cout << "\t MatVecMul - EvalMul... " <<  getCurrentTime() << " line " << i << std::endl;
//         Ciphertext<DCRTPoly> mult = cc->EvalMult(encMatrixRows[i], encVec);

//         // Sum all elements in the vector to simulate dot product
//         std::cout << "\t MatVecMul - Sum vec up... " <<  getCurrentTime() << " line " << i << std::endl;

//         Ciphertext<DCRTPoly> sum = mult;
//         for (size_t j = 1; j < dim; j <<= 1) {
//             auto rotated = cc->EvalAtIndex(mult, j);
//             sum = cc->EvalAdd(sum, rotated);
//         }

//         resultRows[i] = sum;
//     }

//     // Optional: pack the result rows into a single ciphertext
//     // For now return vector of single values
//     return resultRows;
// }



Ciphertext<DCRTPoly> EncryptedVecSumAll(
    CryptoContext<DCRTPoly> cc,
    Ciphertext<DCRTPoly>& encVec,
    size_t len) {
        Ciphertext<DCRTPoly> res = encVec->Clone();
        for (size_t i = 1; i < len; i++){
            res = cc-> EvalAdd(res, cc->EvalRotate(encVec, i));
        }

    return res;
}

// std::vector<Ciphertext<DCRTPoly>> EncryptedMatrixVectorMultiply(
//     CryptoContext<DCRTPoly> cc,
//     const std::vector<Ciphertext<DCRTPoly>>& encMatrix,
//     // std::vector<std::vector<Ciphertext<DCRTPoly>>>& encMatrix,
//     const Ciphertext<DCRTPoly>& encVector,
//     size_t rows
// ) {
    
//     std::vector<Ciphertext<DCRTPoly>> resultRows;

//     for (size_t i = 0; i < rows; i++) {
//         // Ciphertext<DCRTPoly> sum = cc->EvalMult(encMatrix[i][0], encVector);
//         // for (size_t j = 1; j < cols; ++j) {
//         //     auto rotated = cc->EvalAtIndex(encVector, -int(j));  // Rotate to align with matrix column j
//         //     auto prod = cc->EvalMult(encMatrix[i][j], rotated);
//         //     sum = cc->EvalAdd(sum, prod);
//         // }
//         Ciphertext<DCRTPoly> sum = cc->EvalMultAndRelinearize(encMatrix[i], encVector);
//         resultRows.push_back(sum);
//     }

//     std::vector<Ciphertext<DCRTPoly>> result(rows); 
//     for (size_t i = 0; i < rows; i++){
        
//         std::cout << "Here2: " << i << std::endl;
//         result[i] = EncryptedVecSumAll(cc, resultRows[i], cols);
//     }

//     return result;
// }



Ciphertext<DCRTPoly> EncryptedL2Norm(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& encVector,
    size_t len,
    std::vector<double> coeffs,
    PublicKey<DCRTPoly>& publicKey) {
    
     // Step 1: Element-wise square
    auto ct_squared = cc->EvalMultAndRelinearize(encVector, encVector);

    // Step 2: Sum all slots using rotation + EvalAdd
    // auto ct_sum = ct_squared;
    // for (size_t i = 1; i < len; i <<= 1) {
    //     auto rotated = cc->EvalAtIndex(ct_squared, i);
    //     ct_sum = cc->EvalAdd(ct_sum, rotated);
    // }
    auto ct_sum = EncryptedVecSumAll(cc, ct_squared, len);
    
    // Plaintext resultPlain;
    // cc->Decrypt(keyPair.secretKey, ct_sum, &resultPlain);
    // resultPlain->SetLength(1);
    
    // // std::cout << "Summation Result: " << resultPlain << std::endl;


    // std::cout << "Result: " << resultPlain << std::endl;

    // Step 3: Polynomial approximation of sqrt(x)
    // Example: sqrt(x) ≈ 0.5 + 0.5 * x   (VERY rough approx, better ones below)
    // std::vector<double> coeffs = {0.5, 0.5}; // sqrt(x) ≈ 0.5 + 0.5*x

    // std::vector<double> coeffs = {

    //     // The coefficients needs to be generated regarding the range of the input (i.e., the sum of the squared vector).
    //     // This example is for [0.1, 8.0]
    //     2.8680857437039592e-01,
    //     9.3318419097809968e-01,
    //     -2.7693691155390299e-01,
    //     5.8571484180183764e-02,
    //     -6.2904597034056531e-03,
    //     2.6212278310731299e-04,
    // };


    auto sqrt_poly = cc->EvalPoly(ct_sum, coeffs);
    return sqrt_poly;
}


// The version with input vector as a vector of encryption of single values
Ciphertext<DCRTPoly> EncryptedL2Norm(
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& encVector,
    size_t len,
    std::vector<double> coeffs,
    PublicKey<DCRTPoly>& publicKey) {
    
     // Step 1: Element-wise square
    Ciphertext<DCRTPoly> ct_sum = cc->EvalMult(encVector[0], encVector[0]);
    for (size_t i = 1; i < len; i++){
        auto squared = cc->EvalMult(encVector[i], encVector[i]);
        // ct_squared.push_back(squared);
        ct_sum = cc->EvalAdd(ct_sum, squared);
    }

    // auto ct_sum = EncryptedVecSumAll(cc, ct_squared, len);
    

    // std::vector<double> coeffs = {

    //     // The coefficients needs to be generated regarding the range of the input (i.e., the sum of the squared vector).
    //     // This example is for [0.1, 8.0]
    //     2.8680857437039592e-01,
    //     9.3318419097809968e-01,
    //     -2.7693691155390299e-01,
    //     5.8571484180183764e-02,
    //     -6.2904597034056531e-03,
    //     2.6212278310731299e-04,
    // };


    auto sqrt_poly = cc->EvalPoly(ct_sum, coeffs);
    return sqrt_poly;
}





Ciphertext<DCRTPoly> EncryptedSigmoid(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> z,
    std::vector<double> coeffs,
    PublicKey<DCRTPoly>& publicKey) {
    
    // std::vector<double> coeffs = {

    //     // The coefficients needs to be generated regarding the range of the input (i.e., the sum of the squared vector).
    //     // This example is for [0.1, 8.0]
    //       0.5000000000,
    //       0.2469012124,
    //       -0.0000000000,
    //       -0.0174319718,
    //       0.0000000000, 
    //       0.0009451433,
    //       -0.0000000000,
    //       -0.0000220044
    // };
    
    auto sigmoid_poly = cc->EvalPoly(z, coeffs);
    return sigmoid_poly;
};



// Ciphertext<DCRTPoly> EncryptedSigmoid(
//     const CryptoContext<DCRTPoly>& cc,
//     std::vector<Ciphertext<DCRTPoly>> z,
//     KeyPair<DCRTPoly> keyPair){
    
//     std::vector<double> coeffs = {

//         // The coefficients needs to be generated regarding the range of the input (i.e., the sum of the squared vector).
//         // This example is for [0.1, 8.0]
//           0.5000000000,
//           0.2469012124,
//           -0.0000000000,
//           -0.0174319718,
//           0.0000000000, 
//           0.0009451433,
//           -0.0000000000,
//           -0.0000220044
//     };
    
//     auto sigmoid_poly = cc->EvalPoly(z, coeffs);
//     return sigmoid_poly;
// };


Ciphertext<DCRTPoly> EncryptedInverse(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    std::vector<double> coeffs,
    PublicKey<DCRTPoly>& publicKey) {
        
        // TODO: put the coefficiencts for the range of input in the real application.
        // For now it's for range [0.5, 1.5]
        // std::vector<double> coeffs = {
        //     9.234581052762,
        //     -36.542528473571,
        //     80.951217223236,
        //     -109.847286516978,
        //     93.550716808160,
        //     -48.865496386929,
        //     14.324540312660,
        //     -1.805782690271
        // };

        Ciphertext<DCRTPoly> result  = cc->EvalPoly(x, coeffs);
        return result;
}



// function to compute ||w||^2
// The version with input vector as a vector of encryption of single values
Ciphertext<DCRTPoly> EncryptedSquaredNorm(
    CryptoContext<DCRTPoly> cc,
    const Ciphertext<DCRTPoly>& encVector,
    size_t len, 
    PublicKey<DCRTPoly>& publicKey) {

    auto ct_squared = cc->EvalMultAndRelinearize(encVector, encVector);

    auto ct_sum = EncryptedVecSumAll(cc, ct_squared, len);

    return ct_sum;
}


// Function to approximate 1/x using Newton-Raphson method
Ciphertext<DCRTPoly> EncryptedInverseNR(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    std::vector<double> interval,   // interval = {a, b}
    int iter,                       // number of Newton iterations
    PublicKey<DCRTPoly>& publicKey) {

    double a = interval[0];
    double b = interval[1];

    // Compute constants for initial approximation
    double num = 32.0 * (pow(a, 3) + 7 * pow(a, 2) * b + 7 * a * pow(b, 2) + pow(b, 3));
    double num1 = -32.0 * (5 * pow(a, 2) + 14 * a * b + 5 * pow(b, 2));
    double num2 = 32.0 * 8 * (a + b);
    double num3 = -32.0 * 4;

    double den = pow(a, 4) + 28 * pow(a, 3) * b + 70 * pow(a, 2) * pow(b, 2)
                 + 28 * a * pow(b, 3) + pow(b, 4);

    // Polynomial: (num + num1*x + num2*x^2 + num3*x^3) / den
    std::vector<double> coeffs(4);
    coeffs[0] = num / den;
    coeffs[1] = num1 / den;
    coeffs[2] = num2 / den;
    coeffs[3] = num3 / den;

    // Initial approximation y = P(x)
    std::cout << "\t InverseNR - initial P(x)... " <<  getCurrentTime()  << std::endl;
    Ciphertext<DCRTPoly> y = cc->EvalPoly(x, coeffs);

    // Newton-Raphson Iteration: y = y * (2 - x * y)
    for (int i = 0; i < iter; ++i) {
        std::cout << "\t InverseNR - Newton iterration... " <<  getCurrentTime() << " iter " << i << std::endl;
        Ciphertext<DCRTPoly> xy = cc->EvalMult(x, y);
        // Ciphertext<DCRTPoly> two = cc->EvalConstant(x, 2.0);  // constant ciphertext

        // std::vector two (1, 2.0);
        // Plaintext ptTwo = cc->MakeCKKSPackedPlaintext(two);
        // Ciphertext<DCRTPoly> ctTwo = cc->Encrypt(publicKey, ptTwo);
        // Ciphertext<DCRTPoly> diff = cc->EvalSub(ctTwo, xy);     // (2 - x * y)

        Ciphertext<DCRTPoly> diff = cc->EvalSub(2, xy);
        
        y = cc->EvalMult(y, diff);                            // y * (2 - x * y)
    }

    return y;
}


Ciphertext<DCRTPoly> EncryptedInverseTS(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly> x,
    PublicKey<DCRTPoly>& publicKey) {

    // Term 1: (x - 1)
    // Multiplicative Depth: 0
    Ciphertext<DCRTPoly> term1 = cc->EvalSub(x, 1.0);

    // Term 2: (x - 1)^2
    // Multiplicative Depth: 1
    Ciphertext<DCRTPoly> term2 = cc->EvalMultAndRelinearize(term1, term1);
    // cc->ModReduceInPlace(term2);

    // Term 3: (x - 1)^3
    // Multiplicative Depth: 2
    Ciphertext<DCRTPoly> term3 = cc->EvalMultAndRelinearize(term2, term1);
    // cc->ModReduceInPlace(term3);

    // Combine terms to compute the final polynomial: y = 1 - term1 + term2 - term3
    // The final depth is determined by the deepest term (term3).
    Ciphertext<DCRTPoly> result = cc->EvalSub(1.0, term1);
    result = cc->EvalAdd(result, term2);
    result = cc->EvalSub(result, term3);
    std::cout << "\t end of InverseTS " <<  getCurrentTime() << std::endl;

    return result;
}