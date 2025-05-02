from data_loading_and_preprocessing import load_and_preprocess_data
from aco_outlier_detection import aco_outlier_detection
from weighted_resampling import weighted_resample
from evaluation_and_results import evaluate_model

data, labels = load_and_preprocess_data()
outliers = aco_outlier_detection(data)
new_data, new_labels = weighted_resample(data, labels, outliers)
evaluate_model(new_data, new_labels)
