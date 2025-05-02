import numpy as np

def weighted_resample(data, labels, outliers):
    mask = np.ones(len(data), dtype=bool)
    mask[outliers] = False

    clean_data = data[mask]
    clean_labels = labels[mask]

    majority = clean_data[clean_labels == 0]
    minority = clean_data[clean_labels == 1]

    weight = len(majority) // len(minority)
    resampled_minority = np.tile(minority, (weight, 1))
    resampled_labels = np.array([1] * len(resampled_minority))

    combined_data = np.vstack((majority, resampled_minority))
    combined_labels = np.array([0] * len(majority) + list(resampled_labels))

    return combined_data, combined_labels
