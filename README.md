# JointYard

This repository contains a Python package designed for analyzing pose tracking data obtained from software like [Lightning Pose](https://github.com/danbider/lightning-pose).  
The toolkit provides methods for processing and interpreting joint coordinate data to study behavioral patterns.  

JointYard is built to analyze large-scale datasets with minimal supervision, capturing both **spatial and temporal** dynamics of behavior.  
By leveraging **unsupervised learning** and **nonlinear dynamic embeddings** (e.g., VAE), it uncovers complex behavioral hierarchies and transitions over time.  

The system is **scalable** and **automated**, making it ideal for high-throughput behavioral analysis in naturalistic settings.  
Ultimately, this framework serves as a foundation for integrating behavioral and neural data, paving the way for a **comprehensive understanding of brain-behavior relationships** in future studies.  


https://github.com/user-attachments/assets/3f816a22-69ad-4e05-99ee-233bdfc231a8


---

## Installation  

**Tested on:**  
- **Ubuntu 20.04**  
- **RTX 4090**  
- **CUDA 12.3**   

### Steps  
1️. **Create and activate a virtual environment**
```bash
conda create -n JointYard python==3.10
conda activate JointYard
```
2️. **Git clone and Navigate to the project directory**
```bash
git clone https://github.com/thejiyounglee/JointYard.git
cd /path/to/JointYard
```
3️. **Install dependencies and RAPIDS cuML for GPU acceleration**
```bash
pip install -r requirement.txt
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.10.* cuml-cu12==24.10.*
```
> **Note:** If you do not have a **GPU** or cannot use **cuml**, you can modify the import statements  
> to use CPU-based alternatives (e.g., by commenting out PCA, UMAP, HDBSCAN imports).
> changing it to the appropriate version for the environment.
> However, This code is based on the environment written above.
---
## Pipeline Overview  
![스크린샷_2025-03-14_18-01-08](https://github.com/user-attachments/assets/a721dcda-274b-4e0e-ba58-daf8efc46323)
![스크린샷_2025-03-14_18-01-41](https://github.com/user-attachments/assets/df588734-f891-469a-8f61-31135f3b0126)

1️. **Preprocessing**  
   - Load and clean pose-tracking data  
   - Standardize input formats  

2️. **Spectrogram Transformation**  
   - Convert time-series behavior data using Continuous Wavelet Transform (CWT)  

3️. **Nonlinear Embedding**  
   - Extract low-dimensional features using Deep Variational Autoencoder (VAE)  

4️. **Dimensionality Reduction**  
   - Reduce high-dimensional data to 2D using UMAP  

5️. **Spatial Segmentation**  
   - Cluster behavior patterns using HDBSCAN  

6️. **Ethogram Construction**  
   - Compute behavioral transition matrices to analyze movement between states  

7️. **Other Analysis**  
   - Perform traditional statistical tests 
   - Markovian process with transition analysis  
   - SPAEF analysis  

---
## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/thejiyounglee/JointYard/blob/main/LICENSE) file for details.
