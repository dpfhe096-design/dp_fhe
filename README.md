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
| **Ring dimension (N)** | 32,768 | Supports bootstrapping and high-precision computation |
| **Scaling moduli** | {60, 59, 59, …} bits | Automatically generated modulus chain under FLEXIBLEAUTOEXT |
| **Secret key distribution** | `UNIFORM_TERNARY` | Efficient ternary secret keys with low noise |
| **Key switch technique** | `HYBRID` | Memory-efficient relinearization |
| **Scaling technique** | `FLEXIBLEAUTOEXT` | Automatic rescaling with bootstrapping support |
| **Multiplicative depth** | \(28 + d_{\text{boot}}\) | Total depth including internal bootstrapping cost |
| **Levels after bootstrapping** | 28 | Usable computation levels restored per bootstrapping |
| **Bootstrapping level budget** | {4, 4} | Precision allocation for bootstrapping |
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

## Example Usage

This example demonstrates how to run encrypted training on **multiple datasets** (e.g., MNIST and Credit) using OpenFHE.  
The workflow includes three stages: **encryption**, **secure training**, and **decryption**.

```bash
# ------------------------
# Environment Setup
# ------------------------
module load gcc/14.2.0

# Set OpenMP parallelism
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# OpenFHE library path
export LD_LIBRARY_PATH=path to.../openfhe/lib:$LD_LIBRARY_PATH

# ------------------------
# Shared Parameters
# ------------------------
budget1=4
budget2=4
levels=28          
T=1000            
b_theta=5.0
b_lambda=0.001
batch_size=100

# ------------------------
# Dataset Configuration
# ------------------------

# MNIST dataset
dataset_name="mnist"
changeLog="model: clip_train, data=mnist"
lr=0.041
dim_m=50           # Number of features/slots for MNIST
checkpointDir="enc_${dataset_name}"

# Credit dataset (uncomment to use)
# dataset_name="credit"
# changeLog="model: clip_train, data=credit"
# lr=0.06
# dim_m=24
# checkpointDir="enc_${dataset_name}"

# ------------------------
# Paths and Arguments
# ------------------------
args_enc=("$checkpointDir" "$changeLog" $budget1 $budget2 "$dataset_name" $levels)
args_train=("$checkpointDir" "$changeLog" $budget1 $budget2 "$dataset_name" \
            $lr $T $batch_size)

# ------------------------
# Build
# ------------------------
cd path to.../build
make

# ------------------------
# Step 1: Setup & Encryption
# ------------------------
set -e
mkdir -p "$checkpointDir"

echo "====== SETUP & ENCRYPTION ======"
date
./setup_encrypt "${args_enc[@]}"

# ------------------------
# Step 2: Secure Model Training
# ------------------------
echo "====== SECURE MODEL TRAINING ======"
date
./model_clipping "${args_train[@]}"
date

# ------------------------
# Step 3: Decrypt Trained Weights
# ------------------------
echo "====== DECRYPT MODEL WEIGHTS ======"
date
./decrypt_weight "$checkpointDir" "$dim_m"
date

```
## Results 
## Model performance and training time under FHE (AMD EPYC 9534, multi-threaded)
**Note:** \(\epsilon=1, \delta=10^{-5}\)

| Data    | Training Model                        | ACC     | AUC     | 10-threads      | 20-threads      | 30-threads      | 50-threads      |
|---------|--------------------------------------|---------|---------|----------------|----------------|----------------|----------------|
| mnist   | Algo~\ref{fig:trad_DP-GD_fhe} (DP-SGD) | 95.83%  | 98.88%  | 600.1 sec/iter | 338.0 sec/iter | 283.0 sec/iter | 187.2 sec/iter |
| mnist   | Algo~\ref{fig:modified_DP-GD_fhe} (No Clip) | 94.18%  | 98.32%  | 148.2 sec/iter | 93.0 sec/iter  | 86.9 sec/iter  | 58.2 sec/iter  |
| credit  | Algo~\ref{fig:trad_DP-GD_fhe} (DP-SGD) | 78.61%  | 70.60%  | 446.1 sec/iter | 258.7 sec/iter | 227.5 sec/iter | 164.0 sec/iter |
| credit  | Algo~\ref{fig:modified_DP-GD_fhe} (No Clip) | 77.95%  | 71.39%  | 132.2 sec/iter | 79.4 sec/iter  | 70.4 sec/iter  | 53.7 sec/iter  |



