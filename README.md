# POD-DMD-Reduced-Order-Modeling

## Project Overview: Data-Driven Model Reduction for CFD

This repository contains the implementation and comparative analysis of **Proper Orthogonal Decomposition (POD)** and **Dynamic Mode Decomposition (DMD)** for **Reduced Order Modeling (ROM)**. The project's goal is to significantly reduce the computational complexity (degrees of freedom) of a high-dimensional dynamical system while preserving the core physics.

The case study focuses on the **vorticity field** of a **2D Navier-Stokes flow past a cylinder** at a moderate Reynolds number ($\text{Re}=200$), a system characterized by complex vortex shedding (Kármán vortex street).

---

## Repository Contents

| File Name | Method | Description |
| :--- | :--- | :--- |
| `pod_energy_analysis.py` | **POD** | Python script for SVD, calculating singular values, cumulative energy, and plotting energy distribution. |
| `dmd_prediction_error.py` | **DMD** | Python script implementing the time-shifted matrix method, calculating the low-rank operator $\mathbf{\tilde{A}}$, and determining the **prediction error** across various mode ranks. |
| `pod_dmd_full_report.pdf` | **Full Report** | **Crucial Deliverable:** Contains all detailed methodology, convergence plots (Figures 1-11), mode visualizations (POD and DMD), and a full comparative discussion of results. |
| `README.md` | - | This project description and guide. |

---

##  Key Methodologies Demonstrated

### 1. Proper Orthogonal Decomposition (POD)

* **Basis Generation:** Utilized **Singular Value Decomposition (SVD)** on the data snapshot matrix ($\mathbf{U}$) to extract an **energy-optimal, orthogonal spatial basis** (the POD modes, $\mathbf{\Phi}$).
* **Dimensionality Reduction:** Demonstrated that the majority of the system's energy (e.g., **~89%** of cumulative energy) can be captured by only **40 modes** ($r=40$), reducing the state dimension from $n \approx 147,000$ to a low-dimensional space.

### 2. Dynamic Mode Decomposition (DMD)

* **Linear Modeling:** Constructed a **data-driven linear operator** ($\mathbf{\tilde{A}}$) to model the time-evolution of the fluid system ($\mathbf{x}_{k+1} \approx \mathbf{A} \mathbf{x}_k$).
* **Dynamic Modes:** Computed the DMD modes and eigenvalues, revealing **complex conjugate pairs** of eigenvalues, which correspond precisely to the **oscillatory and rotating vortex dynamics** characteristic of the flow.
* **Prediction vs. Compression:** Compared DMD's ability to **predict** out-of-sample snapshots against POD's ability to **compress** in-sample snapshots.

---

## Key Findings and Convergence (Refer to `pod_dmd_full_report.pdf`)

* **POD Convergence:** Relative approximation error ($\frac{||\mathbf{U} - \mathbf{U}_{\text{rank}}||_F}{||\mathbf{U}||_F}$) decreased exponentially with increasing rank, confirming the high efficiency of the POD basis for data compression.
* **DMD Accuracy:** Analysis of the prediction error $\frac{||\mathbf{\Phi}\mathbf{\Lambda}^t \mathbf{b} - \mathbf{x}_{t}||}{||\mathbf{x}_t||}$ showed high accuracy for in-sample snapshots (requiring $\sim 35$ modes for $10\%$ error) but reduced accuracy for out-of-sample prediction.

---

## Data and Reproducibility

The raw fluid dynamics data file (`cyldata6h.csv`) is **not included** in this repository due to its large size and proprietary nature.

The provided Python scripts are fully functional. To reproduce the analysis and plots, the raw data must be loaded into the `U` variable (where $U$ is the state matrix). **The complete analysis, visualizations, and conclusions are contained within the `pod_dmd_full_report.pdf` file.**
