# ESMethods

Solving Reinforcement Learning tasks on the OpenAI Gym platform, using ES Methods to train the action policy.

Tasks include:
- CartPole
- MountainCarContinuous
- Acrobot

The videos of the trained policies are in the videos folder.

Implemented and compared different gradient estimation methods: 
- Monte Carlo algorithms
  - Gaussian Sampling
  - Gaussian Orthogonal Matrices
  - Random Hadamard Matrices
  - Givens Random Rotations
- Regression-based algorithms
  - Ridge
  - Lasso
  - LP-decoding

Implemented and compared different control variate term mechanisms: 
- vanilla
- antithetic
- forward-fd
