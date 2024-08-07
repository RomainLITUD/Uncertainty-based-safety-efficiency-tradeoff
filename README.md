# Uncertainty-based-safety-efficiency-tradeoff-for-AVs

- This is the paper's code **How Far Ahead Should Autonomous Vehicles Start Resolving Predicted Conflicts? Exploring Uncertainty-Based Safety-Efficiency Trade-Off**

- The paper is available online via IEEE Transactions on Intelligent Transportation Systems [link (click)](https://ieeexplore.ieee.org/abstract/document/10528252)

### Requirements:

* Python = 3.9
* PyTorch ≥ 1.11
* Shapely = 1.8.5
* Zarr

### Data Preparation

* The used INTERACTION dataset can be found by [Interaction Webpage](https://interaction-dataset.com/).
* The data processing process can be found on [this page (click)](https://github.com/RomainLITUD/UQnet-arxiv)

### Model Training

* Run the `TrainingModels.ipynb` to train the proposed model.
* The pre-trained model is in the folder `pretrain`. The encoder and decoder can be directly used.

### Trade-off relationship studies.

-The final published version uses a different hybrid method (both IDM simulation and geometry computation), will be updated soon.
