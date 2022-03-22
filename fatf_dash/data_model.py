import joblib
import numpy as np
import os

# Data imports
from fatf_dash.census import map_s_i, census_names


this_dir = os.path.dirname(os.path.abspath(__file__))
data_model_path = os.path.abspath(os.path.join(this_dir, '../_data_model'))


clf = joblib.load(os.path.join(data_model_path, 'log_reg.joblib'))


original_data = np.load(os.path.join(data_model_path, 'adult_num.pkl.npy'))
original_ground_truth = np.load(os.path.join(data_model_path, 'adult_num_gt.pkl.npy'))
#
indices = (list(range(0, 100)) +
           list(range(1100, 1200)) +
           list(range(5100, 5200)) +
           list(range(7100, 7200)))
#
data = original_data[indices, :]
ground_truth = original_ground_truth[indices]
#
## 222                          0: sex
wronged_1 = np.array([[32, 4, 173314, 8, 11, 2, 9, 0, 3, 0, 0, 0, 60, 38]])
wronged_1_gt = np.array([1])  # 0
## 333                          9: race
wronged_2 = np.array([[34, 2, 340458, 2, 8, 4, 0, 4, 1, 0, 0, 0, 40, 38]])
wronged_2_gt = np.array([1]) #  0
#
data = np.concatenate([data, wronged_1, wronged_2], axis=0)
ground_truth = np.concatenate([ground_truth, wronged_1_gt, wronged_2_gt])


original_predictions = clf.predict(original_data)
predictions = original_predictions[indices]


# Column indices of maps
map_indices = dict()
for feature_name in map_s_i:
    idx = census_names.index(feature_name)
    map_indices[idx] = feature_name
