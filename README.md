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
| **Scheme** | CKKS  | Approximate arithmetic for encrypted real numbers |
| **Ring dimension (N)** | 32,768 / 131,072 | Supports bootstrapping and high-precision computation |
| **Scaling moduli** | {60, 59, 59, …} bits | Automatically generated modulus chain under FLEXIBLEAUTOEXT |
| **Secret key distribution** | `UNIFORM_TERNARY` | Efficient ternary secret keys with low noise |
| **Key switch technique** | `HYBRID` | Memory-efficient relinearization |
| **Scaling technique** | `FLEXIBLEAUTOEXT` | Automatic rescaling with bootstrapping support |
| **Multiplicative depth** | '28 + depth_{bootstrap}' | Total depth including internal bootstrapping cost |
| **Levels after bootstrapping** | 28 | Usable computation levels restored per bootstrapping |
| **Bootstrapping level budget** | {4, 4} | Precision allocation for bootstrapping |
| **Security level** |  ~128-bit (at N = 131,072) | Full 128-bit security achieved with larger ring dimension |

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
#!/bin/bash
# ------------------------
# SLURM Job Settings
# ------------------------
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=256gb              # memory allocation

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
T=200            
b_theta=5.0
b_lambda=0.001
batch_size=100

# ------------------------
# Dataset Configuration
# ------------------------

# MNIST dataset
dataset_name="mnist"
changeLog="model: clip_train, data=mnist"
lr=0.03
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
## Model performance and training time under FHE 
---
**Train Config:** `epsilon=1`, `delta=1e-5`, `T=200`, `lr=0.03` (mnist), `lr=0.06` (credit)

### Machine=AMD EPYC 9534 (Multi-threaded), RingDim = 32768
**Bootstrapping:** Our method – every 3 iterations; DP-SGD – every iteration

| Data   | Training Model | ACC     | AUC     | 10-threads      | 20-threads      | 30-threads      | 50-threads      |
|--------|----------------|---------|---------|----------------|----------------|----------------|----------------|
| mnist  | DP-SGD         | 93.62%  | 98.21%  | 600.1 sec/iter | 338.0 sec/iter | 283.0 sec/iter | 187.2 sec/iter |
| mnist  | Our method     | 93.99%  | 98.22%  | 148.2 sec/iter | 93.0 sec/iter  | 86.9 sec/iter  | 58.2 sec/iter  |
| credit | DP-SGD         | 78.00%  | 68.88%  | 446.1 sec/iter | 258.7 sec/iter | 227.5 sec/iter | 164.0 sec/iter |
| credit | Our method     | 77.99%  | 68.69%  | 132.2 sec/iter | 79.4 sec/iter  | 70.4 sec/iter  | 53.7 sec/iter  |

---

### Machine=AMD EPYC 7502 (Multi-threaded), RingDim = 131072
**Bootstrapping:** Our method – every iteration; DP-SGD – twice every iteration

| Data   | Training Model | ACC     | AUC     | 10-threads        | 20-threads        | 30-threads        | 50-threads        |
|--------|----------------|---------|---------|------------------|------------------|------------------|------------------|
| mnist  | DP-SGD         | 93.77%  | 98.20%  | 7996.8 sec/iter  | 4240.0 sec/iter  | 3504.4 sec/iter  | 1813.5 sec/iter  |
| mnist  | Our method     | 93.97%  | 98.20%  | 566.7 sec/iter   | 442.2 sec/iter   | 409.8 sec/iter   | 350.5 sec/iter   |
| credit | DP-SGD         | 77.96%  | 69.18%  | 8028.1 sec/iter  | 3910.5 sec/iter  | 3236.3 sec/iter  | 1843.3 sec/iter  |
| credit | Our method     | 77.99%  | 68.85%  | 521.9 sec/iter   | 449.2 sec/iter   | 394.1 sec/iter   | 343.1 sec/iter   |

---

### Additional Results (Rebuttal Only)

## Learning Rate Tuning on MNIST

<p align="center">
  <img src="lr_tuning_training/noclip_model.png" alt="Algorithm 5 (with barrier)" width="45%"/>
  <img src="lr_tuning_training/clipping_model.png" alt="Algorithm 4 (no barrier)" width="45%"/>
</p>

<p align="center">
  <em>
  Left: Algorithm 5 (with barrier) remains stable at η = 0.3 and converges faster (~40 iterations).  
  Right: Algorithm 4 (no barrier) at η = 0.15 shows slow convergence (~100 iterations), while for η > 0.15 the training becomes unstable and diverges (loss = NaN).
  </em>
</p>



