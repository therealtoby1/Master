# Master Thesis

The topic of this Master thesis is the `<u>` Data based modeling of Integro-Differential Equations `</u>`. Exemplary Equations can be found in the context of cell populations and/or crystal population models.

First such equations are solved numerically (using python). The solution is a population model n(m,t). In the next step it is attempted to recreate the system dynamics using a model-based approach.
This can be realized using **neural networks**, **DMD** or other Methods such as **Gaussian Processes**. the "real" DGL is then simply used for verification.

Then it is attempted to design an Observer for the model, where sampled ouput data y is provided. Observers suitable for this task include **Unscented Kalman Filters** or **Moving Horizon Estimation**

The 04/07/2025 marks the start of my Master Thesis. Therefore i will write weekly summaries:

---

*07/07-11/07*

- [Easier bioreactor model](https://github.com/therealtoby1/Master/blob/main/Cell_growth_easy_Model.ipynb) in the form of a simple ODE for the dynamics to test out Data based modeling approaches as well as observer designs. Following things have been implemented there:

  - simple Neural Networks with linear hidden layers (no PINNs yet, relatively good fit with little training sets)
  - DMD and Hankel DMD (Hankel because n=2-->low dimension, however bad fit for both)
  - Gaussian Process (work in progress). Here the choice of the kernel plays a crucial part. I tried combining the classic RBF Kernel with an exponentially decaying kernel (for the substrate), however this resulted in the decaying part dominating over the RBF part, thus the covariance being highest when substrate concentrations are similar. Because of this the set of Functions that could be generated also falsly indicated the biomass concentration. The issue was using a combined kernel. maybe better results could be achieved through multi-ouput-GPs where each output gets a seperate kernel or simply leaving out the exponential part

  and for the Observer :

  - Unscented Kalman Filter (using filterpy)

---
