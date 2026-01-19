# DP-HE-Training: Differentially Private Logistic Regression in Homomorphic Encryption

## Overview

This project demonstrates a high-performance system for training a logistic regression model using **Homomorphic Encryption (HE)** combined with **Differential Privacy (DP)**. It uses the **OpenFHE** library to perform all computations on encrypted data, ensuring that the raw data, model parameters, and intermediate gradients are never exposed to the server.

By combining HE and DP, this project provides two layers of protection:

1.  **Homomorphic Encryption (CKKS):** Guarantees that all data and model parameters are encrypted during computation. The server *cannot* see the data.

2.  **Differential Privacy:** Provides formal, mathematical proof that the final, trained model does not reveal information about any single individual in the training dataset.

At its core, the system securely executes a mini-batch gradient descent algorithm, including linear hypothesis, non-linear sigmoid approximation, and weight updates, all within the encrypted domain.

## Cryptographic Configuration

| Parameter | Value | Description |
|------------|--------|-------------|
| **Scheme** | CKKS (RNS variant) | Approximate arithmetic for encrypted real numbers |
| **Ring dimension (N)** | 32,768 | Supports bootstrapping and high precision |
| **Scaling moduli** | {60, 59, 59, …} bits | Defines precision at each level |
| **Secret key distribution** | `UNIFORM_TERNARY` | Efficient and noise-minimizing |
| **Key switch technique** | `HYBRID` | Memory-efficient relinearization |
| **Scaling technique** | `FLEXIBLEAUTOEXT` | Enables automatic rescaling and bootstrapping |
| **Multiplicative depth** | ~30+ | Provides approximately 26 usable computation levels after each bootstrapping operation |
| **Security level** | ~128-bit (manual) | Achieved with N = 32,768 and 59-bit primes |

---

## Bootstrapping Setup

Bootstrapping is configured to **refresh ciphertexts** and restore noise capacity, enabling unlimited homomorphic operations.

```cpp
std::vector<uint32_t> levelBudget = {4, 4};
parameters.SetRingDim(32768);
parameters.SetScalingModSize(59);
parameters.SetFirstModSize(60);
parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
parameters.SetKeySwitchTechnique(HYBRID);
```

## Example
```bash
changeLog="full_train mnist nosie_1 batch=100, log last 100 iter weights"
budget1=4
budget2=4
dataset_name="mnist"
save_path="enc_"$dataset_name
levels=26
lr=0.041
T=1000
b_theta=5.0
b_lambda=0.001
batch_size=100
args_enc=("$save_path" "$changeLog" $budget1 $budget2 "$dataset_name" $levels)
args_train=("$save_path" "$changeLog" $budget1 $budget2 "$dataset_name" $lr $T $b_theta $b_lambda $batch_size)

set -e  
mkdir -p "$save_path"
./setup_encrypt.cpp "${args_enc[@]}"
./model_no_clipping.cpp "${args_train[@]}"
```
## Results 
RAM=47G, machine: rhel8,x86_64,Zen,EPYC-9534 x 50 threads
|dataset|accuracy|AUROC|Training_time|RAM|machine|
|------------|--------|-------------|----------------|--------|-----------------------|
|mnist|95.1%|96%|19.5 hrs|47G|rhel8,x86_64,Zen,EPYC-9534 x 50 threads|
|adult|95%|95%|19.5 hrs|47G|rhel8,x86_64,Zen,EPYC-9534 x 50 threads|


