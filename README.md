# ACO Outlier Detection with Weighted Resampling

This project uses Ant Colony Optimization (ACO) to detect outliers in the Credit Card Fraud dataset and applies weighted resampling techniques to handle class imbalance. It includes data loading, preprocessing, evaluation, and visualization.

## Files Overview

| **File Name**                     | **Description**                                                                                                                                         |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `aco_outlier_detection.py`         | Implements the main logic for the **Ant Colony Optimization (ACO)** algorithm to detect outliers in the dataset. This file includes the pheromone update rules and guides ants to find outlier nodes effectively. |
| `data_loading_and_preprocessing.py`| Handles loading and preprocessing the **Credit Card Fraud Dataset**. It reads the dataset, scales features using MinMaxScaler, and separates the labels from the features. |
| `evaluation_and_results.py`        | Evaluates the performance of the ACO-based outlier detection. It calculates precision, recall, F1-score, and accuracy based on detected outliers and compares them with ground truth. Includes visualization of results. |
| `main.py`                          | The entry point that integrates the entire pipeline, from data loading to evaluation. It calls functions for outlier detection, resampling, and evaluation. |
| `weighted_resampling.py`           | Implements **weighted resampling** for handling class imbalance. It balances the dataset by over-sampling the minority class and under-sampling the majority class based on weights. |



## 🔧 Requirements

- Python 3.7+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required packages with:

```bash
pip install -r requirements.txt
```

📊 Dataset
The dataset should be placed in the root directory and named:
```
python main_pipeline.py
```

📌 Notes

This approach avoids synthetic sample generation and focuses on enhancing the presence of real minority samples.

The ACO algorithm considers local distances and pheromone trails to identify sparse and suspicious samples.

Weighted resampling increases the chance of selecting underrepresented but important samples for training.
