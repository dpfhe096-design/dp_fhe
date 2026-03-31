### Additional Results (Rebuttal Only)

## Learning Rate Tuning on MNIST

<p align="center">
  <img src="noclip_model.png" alt="Algorithm 5 (with barrier)" width="45%"/>
  <img src="clipping_model.png" alt="Algorithm 4 (no barrier)" width="45%"/>
</p>

<p align="center">
  <em>
  Left: Algorithm 5 (with barrier) remains stable at η = 0.3 and converges faster (~40 iterations).  
  Right: Algorithm 4 (no barrier) at η = 0.15 shows slow convergence (~100 iterations), while for η > 0.15 the training becomes unstable and diverges (loss = NaN).
  </em>
</p>
