[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkielo3/inekf_demo/blob/main/demo.ipynb)

Single file implementation of an IEKF with Landmarks in Python, directly ported from https://github.com/RossHartley/invariant-ekf

# Definitions

Equation and section pointers to: [Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation](https://arxiv.org/pdf/1904.09251) - please make a pull request with any corrections.

# State Definitions
![](imgs/definitions.png)

# Propagate Step
![](imgs/update.png)

# Correction Step
![](imgs/correct.png)