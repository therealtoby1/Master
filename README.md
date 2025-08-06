# Master Thesis

The topic of this Master thesis is the `<u>` Data based modeling of Integro-Differential Equations `</u>`. Exemplary Equations can be found in the context of cell populations and/or crystal population models.

First such equations are solved numerically (using python). The solution is a population model n(m,t). In the next step it is attempted to recreate the system dynamics using a model-based approach.
This can be realized using **neural networks**, **DMD** or other Methods such as **Gaussian Processes**. the "real" DGL is then simply used for verification.

Then it is attempted to design an Observer for the model, where sampled output data y is provided. Observers suitable for this task include **Unscented /Extended Kalman Filters** or **Moving Horizon Estimation**

Since Training the models may take some time, i will provide the saved models too. i name them according to their layer setup, e.g.  if there is a "2x64x64x2" in the filename this would indicate that to load this model you would need an input vector of dim2, and it gets passed through 2 hidden layers of size 64 and output a vector (or tensor) of size 2 again.

The 04/07/2025 marks the start of my Master Thesis. Therefore i will write weekly summaries:

---

*07/07-11/07*

- [Easier bioreactor model](https://github.com/therealtoby1/Master/blob/main/Cell_growth_easy_Model.ipynb) in the form of a simple ODE for the dynamics to test out Data based modeling approaches as well as observer designs. Following things have been implemented there:

  - simple Neural Networks with linear hidden layers (no PINNs yet, relatively good fit with little training sets)
  - DMD and Hankel DMD (Hankel because n=2-->low dimension, however bad fit for both)
  - Gaussian Process (work in progress). Here the choice of the kernel plays a crucial part. I tried combining the classic RBF Kernel with an exponentially decaying kernel (for the substrate), however this resulted in the decaying part dominating over the RBF part, thus the covariance being highest when substrate concentrations are similar. Because of this the set of Functions that could be generated also falsly indicated the biomass concentration. The issue was using a combined kernel. maybe better results could be achieved through multi-ouput-GPs where each output gets a seperate kernel or simply leaving out the exponential part

  and for the Observer :

  - Unscented Kalman Filter (using filterpy)

For the start i try to get a basic understanding of the different methods and keep the input constant (autonomous system)

---

*14/07-18/07*

- Improving the [Easier bioreactor model](https://github.com/therealtoby1/Master/blob/main/Cell_growth_easy_Model.ipynb)
  - Finishing the GP- for modeling the System with and without time as an additional input parameter
  - Implementing a Neural ODE to better catch the actual dynamics of the system--> best fit so far even when trained on one dataset only
  - Testing Unscented Kalman Filter by providing noise measurements
- Possible improvements:
  - learning system behaviour with input (Dilution rate not only constant anymore)
- 18/07: Meeting with P.Jerono

---

*21/07-25/07*
Still working on the "easy" system model. A big part of this week was retraining the models because of errors i made previously.

1. The datasets i gathered for training were all on the same trajectory-->not much information outside of this one trajectory could be obtained
2. NODE optimization does take time since we are solving an ode for each iteration. Before i didnt let it run long enough, but now , running it for about 1.5h lead to a pretty good function

- Implementation of the Extended Kalman Filter for the original system and the Neural networks (GP is work in Progress) using autograd for the NN.
- Reading into the literature on Moving Horizon estimation as a third type of observer

---

*28/07-31/07*
- Implementing the EKF for all models completed
- Moving Horizon optimization also coded
- Using inputs for the dilution rate in the "easy" system model and watching which models were best suited for the approximation
  - NN performed best (fast training and best fit)
  - attempt at making a "sliding window training" for the models ... was not as good as i initially expected it to be 
  - Node was also good, but training was very time expensive
  - GP couldnt take all the training datasets--> complexity O(n^3) for inverting -->only trained on subset, but still very good fit
  - Patrick Kidger pHD thesis--> continous normalizing flows lesen
  - Neural Operator (apparently better suited for pdes than Neural ODEs)

---

**02/08-08/08*
Thinking about creating a new branch called "to_torch" since this looks like its going to be a very coding heavy week and i will have to make quite a lot of changes like moving system strucures to torch to ensure optimization works. 
1. Move original System and GP to torch (Saturday)-->done
2. adapt changes to EKF (and UKF) (Sunday+Monday/Tuesday) and apply continous EKF to the systems... -->done and also improved ODEfunc to take scalar inputs 
3. fix MHE (Tuesday/Wednesday/Thursday) and test for more samples...    -->done but tuning on saturday
4. compare Jacobians of the system between NN and original system(when using EKF) (Thursday/Friday)-->done
5. Testing the Sensitivity of Q (Q/R) (Saturday)-->do sometime when i got time... before that tho: look at DMD again (Meurer lecture notes...)




Issues of the previous week:
-  MHE : currently the computational graph of x0 is lost when transferring the structure to numpy --> backprop doesnt "see" x0 after it predicts the next step in the moving horizon
--> might have to rewrite the original system to torch  as well as the GP, because otherwise it would be too much effort... 

