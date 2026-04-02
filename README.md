# GluEx: Explainable AI Tools for Deep Learning-Based Glucose Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-FFD43B.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9.20-blue?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/release/python-3920/)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://nbviewer.org/github/calza3000/gluex/blob/main/quick_demo.ipynb)
<!-- [![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore®-00629B)](https://doi.org/xx.yyyyy/zzzzzzzz) -->

<!-- [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.10.0-D00000?logo=keras&logoColor=D00000)](https://keras.io/) -->

This repository provides a suite of explainability tools for deep learning-based glucose forecasting models. It includes utilities for impulse response analysis, partial dependence plots (PDPs), and counterfactual explanations tailored for time-series forecasting.

## Key Contents

-   **`Models/`**: Pre-trained models including `CNN.keras`, `LSTM.keras`, `CNN-LSTM.keras`, `PhyNet.keras`, and `CNN-Transformer.tf/`. Each model takes continuous glucose monitoring (CGM), insulin, and carbohydrate (CHO) data from the previous 90 minutes and predicts the CGM trajectory for the next 90 minutes. Architectural details are available in the associated paper.
-   **`Saved_data/`**: Exported HTML visualizations of impulse responses and PDPs for the example models.
-   **`gluex/`**: A Python package containing the project's core utilities, such as model loading, preprocessing helpers, and explainability functions (`PartialDependencePlot`, `impulse_response`, `counterfactual_analysis`).
-   **`quick_demo.ipynb`**: A Jupyter notebook that demonstrates how to load a model and generate explainability plots.
-   **`requirements.txt`**: A list of dependencies required to run the project.
-   **`CITATION.md`**: Citation information for referencing this work.
-   **`LICENSE`**: The license for this project.

## Quick Start

### Prerequisites

-   **Python 3.9.20** 
-   **Conda** (or another environment manager)

### Installation

1.  **Create and activate a Conda environment:**

    ```powershell
    conda create -n myenv python=3.9.20
    conda activate myenv
    ```

2.  **Clone the repository:**

    ```powershell
    git clone https://github.com/calza3000/gluex.git
    cd gluex
    ```

3.  **Install the required dependencies:**

    ```powershell
    pip install -r requirements.txt
    ```

4.  **Launch the Jupyter notebook:**
    Open and run the cells in `quick_demo.ipynb` to see the tools in action.

## Running the Quick Demo

The `quick_demo.ipynb` notebook provides a hands-on demonstration of the `gluex` package using mock data. It covers:

-   Loading a pre-trained model from the `Models/` directory.
-   Generating mock inputs for `x_test` (which you can replace with your own data for real-world analysis).
-   Creating explainability plots with `PartialDependencePlot` and `impulse_response`.
-   Running a counterfactual analysis using the `counterfactual_analysis` function, which is built on the [DiCE library](https://github.com/interpretml/DiCE).

To get started, open the notebook in Jupyter Lab or a similar environment and execute the cells sequentially.

## License & Citation

This project is licensed under the MIT License. See the `LICENSE` file for more details. 

If you use these tools or models in your research, please cite this repository:

```bibtex
@Article{CalzavaraPhyNet,
  title ="A Physiology-Constrained Monotonic Neural Network for Safe and Explainable Blood Glucose Forecasting",
  author={Calzavara, Andrea and Prendin, Francesco and Cappon, Giacomo and Del Favero, Simone and Facchinetti, Andrea},
  year = "2026",
  journal = "IEEE Transactions on Biomedical Engineering - EMBC 2025 Top Papers",
  doi = "Under review"
}
```

For more citation formats, see `CITATION.md`.

## Contact

-   **Author**: Andrea Calzavara
-   **Email**: calzavaraa@dei.unipd.it

