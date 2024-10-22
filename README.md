# Deep Fusion Model(DFM)

This provides all the source code and data used in Deep Fusion of Discrete-Time and Continuous-Time Models for Long-Term Prediction of Chaotic Dynamical Systems. The code and instructions for usage are currently being organized.

## device
#### system
ubuntu 20.04

#### CPU
11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz 2.50 GHz

Because of the presence of matrix inverses, different processors and library versions may require fine-tuning of the training parameters

## files
#### lorenz_without_noise
noise-free data prediction

#### lorenz_noise_60
noise data(60dB) prediction

#### lorenz_noise_40
noise data(60dB) prediction

#### double_pendulum
double_pendulum simulation data prediction

#### double_pendulum_ex3
double_pendulum experiment data prediction


pinn-sr.py for train continous-time model
ng-rc.py for train decrete-time model
DFM.py for deep fusion model
