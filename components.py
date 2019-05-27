import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import sklearn.linear_model
import joblib

clf = joblib.load('./adult_data/log_reg.joblib')

indices = (list(range(0, 100)) +
           list(range(1100, 1200)) +
           list(range(5100, 5200)) +
           list(range(7100, 7200)))

original_data = np.load('./adult_data/adult_num.pkl.npy')
original_ground_truth = np.load('./adult_data/adult_num_gt.pkl.npy')
original_predictions = clf.predict(original_data)

data = original_data[indices, :]
ground_truth = original_ground_truth[indices]
predictions = original_predictions[indices]


# 222                                                    0: sex
wronged_1 = np.array([[32, 4, 173314, 8, 11, 2, 9, 0, 3, 0, 0, 0, 60, 38]])
wronged_1_gt = np.array([1])  # 0
# 333                                                9: race
wronged_2 = np.array([[34, 2, 340458, 2, 8, 4, 0, 4, 1, 0, 0, 0, 40, 38]])
wronged_2_gt = np.array([1]) #  0

data = np.concatenate([data, wronged_1, wronged_2], axis=0)
ground_truth = np.concatenate([ground_truth, wronged_1_gt, wronged_2_gt])

map_i_s = {
    'workclass': {0: 'Federal-gov',
                  1: 'Local-gov',
                  2: 'Private',
                  3: 'Self-emp-inc',
                  4: 'Self-emp-not-inc',
                  5: 'State-gov',
                  6: 'Without-pay'},
    'education': {0: '10th',
                  1: '11th',
                  2: '12th',
                  3: '1st-4th',
                  4: '5th-6th',
                  5: '7th-8th',
                  6: '9th',
                  7: 'Assoc-acdm',
                  8: 'Assoc-voc',
                  9: 'Bachelors',
                  10: 'Doctorate',
                  11: 'HS-grad',
                  12: 'Masters',
                  13: 'Preschool',
                  14: 'Prof-school',
                  15: 'Some-college'},
    'marital-status': {0: 'Divorced',
                       1: 'Married-AF-spouse',
                       2: 'Married-civ-spouse',
                       3: 'Married-spouse-absent',
                       4: 'Never-married',
                       5: 'Separated',
                       6: 'Widowed'},
    'occupation': {0: 'Adm-clerical',
                   1: 'Armed-Forces',
                   2: 'Craft-repair',
                   3: 'Exec-managerial',
                   4: 'Farming-fishing',
                   5: 'Handlers-cleaners',
                   6: 'Machine-op-inspct',
                   7: 'Other-service',
                   8: 'Priv-house-serv',
                   9: 'Prof-specialty',
                   10: 'Protective-serv',
                   11: 'Sales',
                   12: 'Tech-support',
                   13: 'Transport-moving'},
    'relationship': {0: 'Husband',
                     1: 'Not-in-family',
                     2: 'Other-relative',
                     3: 'Own-child',
                     4: 'Unmarried',
                     5: 'Wife'},
    'race': {0: 'Amer-Indian-Eskimo',
             1: 'Asian-Pac-Islander',
             2: 'Black',
             3: 'Other',
             4: 'White'},
    'sex': {0: 'Female', 1: 'Male'},
    'native-country': {0: 'Cambodia',
                       1: 'Canada',
                       2: 'China',
                       3: 'Columbia',
                       4: 'Cuba',
                       5: 'Dominican-Republic',
                       6: 'Ecuador',
                       7: 'El-Salvador',
                       8: 'England',
                       9: 'France',
                       10: 'Germany',
                       11: 'Greece',
                       12: 'Guatemala',
                       13: 'Haiti',
                       14: 'Holand-Netherlands',
                       15: 'Honduras',
                       16: 'Hong',
                       17: 'Hungary',
                       18: 'India',
                       19: 'Iran',
                       20: 'Ireland',
                       21: 'Italy',
                       22: 'Jamaica',
                       23: 'Japan',
                       24: 'Laos',
                       25: 'Mexico',
                       26: 'Nicaragua',
                       27: 'Outlying-US(Guam-USVI-etc)',
                       28: 'Peru',
                       29: 'Philippines',
                       30: 'Poland',
                       31: 'Portugal',
                       32: 'Puerto-Rico',
                       33: 'Scotland',
                       34: 'South',
                       35: 'Taiwan',
                       36: 'Thailand',
                       37: 'Trinadad&Tobago',
                       38: 'United-States',
                       39: 'Vietnam',
                       40: 'Yugoslavia'},
    'income': {0: '<=50K', 1: '>50K'}
}

map_s_i = {
    'workclass': {'Federal-gov': 0,
                  'Local-gov': 1,
                  'Private': 2,
                  'Self-emp-inc': 3,
                  'Self-emp-not-inc': 4,
                  'State-gov': 5,
                  'Without-pay': 6},
    'education': {'10th': 0,
                  '11th': 1,
                  '12th': 2,
                  '1st-4th': 3,
                  '5th-6th': 4,
                  '7th-8th': 5,
                  '9th': 6,
                  'Assoc-acdm': 7,
                  'Assoc-voc': 8,
                  'Bachelors': 9,
                  'Doctorate': 10,
                  'HS-grad': 11,
                  'Masters': 12,
                  'Preschool': 13,
                  'Prof-school': 14,
                  'Some-college': 15},
    'marital-status': {'Divorced': 0,
                       'Married-AF-spouse': 1,
                       'Married-civ-spouse': 2,
                       'Married-spouse-absent': 3,
                       'Never-married': 4,
                       'Separated': 5,
                       'Widowed': 6},
    'occupation': {'Adm-clerical': 0,
                   'Armed-Forces': 1,
                   'Craft-repair': 2,
                   'Exec-managerial': 3,
                   'Farming-fishing': 4,
                   'Handlers-cleaners': 5,
                   'Machine-op-inspct': 6,
                   'Other-service': 7,
                   'Priv-house-serv': 8,
                   'Prof-specialty': 9,
                   'Protective-serv': 10,
                   'Sales': 11,
                   'Tech-support': 12,
                   'Transport-moving': 13},
    'relationship': {'Husband': 0,
                     'Not-in-family': 1,
                     'Other-relative': 2,
                     'Own-child': 3,
                     'Unmarried': 4,
                     'Wife': 5},
    'race': {'Amer-Indian-Eskimo': 0,
             'Asian-Pac-Islander': 1,
             'Black': 2,
             'Other': 3,
             'White': 4},
    'sex': {'Female': 0, 'Male': 1},
    'native-country': {'Cambodia': 0,
                       'Canada': 1,
                       'China': 2,
                       'Columbia': 3,
                       'Cuba': 4,
                       'Dominican-Republic': 5,
                       'Ecuador': 6,
                       'El-Salvador': 7,
                       'England': 8,
                       'France': 9,
                       'Germany': 10,
                       'Greece': 11,
                       'Guatemala': 12,
                       'Haiti': 13,
                       'Holand-Netherlands': 14,
                       'Honduras': 15,
                       'Hong': 16,
                       'Hungary': 17,
                       'India': 18,
                       'Iran': 19,
                       'Ireland': 20,
                       'Italy': 21,
                       'Jamaica': 22,
                       'Japan': 23,
                       'Laos': 24,
                       'Mexico': 25,
                       'Nicaragua': 26,
                       'Outlying-US(Guam-USVI-etc)': 27,
                       'Peru': 28,
                       'Philippines': 29,
                       'Poland': 30,
                       'Portugal': 31,
                       'Puerto-Rico': 32,
                       'Scotland': 33,
                       'South': 34,
                       'Taiwan': 35,
                       'Thailand': 36,
                       'Trinadad&Tobago': 37,
                       'United-States': 38,
                       'Vietnam': 39,
                       'Yugoslavia': 40},
    'income': {'<=50K': 0, '>50K': 1}
}

census_names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
]

# Column indices of maps
map_indices = dict()
for feature_name in map_s_i:
    idx = census_names.index(feature_name)
    map_indices[idx] = feature_name


def generate_table(max_rows=10):
    indices = np.random.choice(data.shape[0], max_rows, replace=False)
    sample = data[indices, :]
    sample_gt = ground_truth[indices]

    header = [html.Tr([html.Th(col) for col in census_names])]

    body = []
    for d_row, gt_row in zip(sample, sample_gt):
        row = []
        for i in range(sample.shape[1]):
            if i in map_indices:
                value = map_i_s[map_indices[i]][d_row[i]]
            else:
                value = d_row[i]
            row.append(html.Td(value))
        row.append(map_i_s['income'][gt_row])
        body.append(html.Tr(row))

    return html.Table(
        children=(header + body),
        style={'font-size': '90%', 'width': '100%', 'textAlign': 'center', 'align': 'center'}
    )

def datapoint_selection():
    header0 = [html.Tr([html.Th(census_names[i]) for i in range(7)])]
    header1 = [html.Tr([html.Th(census_names[i]) for i in range(7, data.shape[1])])]

    row = []
    for i in range(7):
        feature_name = census_names[i]

        if i in map_indices:
            cat_dict = [{'label': s, 'value': i} for s, i in map_s_i[feature_name].items()]
            # categorical -> radio
            elem = dcc.Dropdown(
                options=cat_dict,
                value='',
                placeholder=feature_name,
                id='{}_input'.format(feature_name))
        else:
            # numerical -> input field
            elem = dcc.Input(
                placeholder=feature_name,
                type='number',
                value='',
                id='{}_input'.format(feature_name))
        row.append(html.Td(elem))
    body0 = [html.Tr(row)]

    row = []
    for i in range(7, data.shape[1]):
        feature_name = census_names[i]

        if i in map_indices:
            cat_dict = [{'label': s, 'value': i} for s, i in map_s_i[feature_name].items()]
            # categorical -> radio
            elem = dcc.Dropdown(
                options=cat_dict,
                value='',
                placeholder=feature_name,
                id='{}_input'.format(feature_name))
        else:
            # numerical -> input field
            elem = dcc.Input(
                placeholder=feature_name,
                type='number',
                value='',
                id='{}_input'.format(feature_name))
        row.append(html.Td(elem))
    body1 = [html.Tr(row)]

    return html.Table(
        children=(header0 + body0 + header1 + body1),
        style={'font-size': '90%', 'width': '100%', 'textAlign': 'center', 'align': 'center'}
    )

def random_point():
    index = np.random.choice(data.shape[0])
    data_point = data[index, :]

    out = []
    for i in range(data_point.shape[0]):
        if i in map_indices:
            feature_name = map_indices[i]
            #out.append(map_i_s[feature_name][data_point[i]])
            out.append(data_point[i])
        else:
            out.append(data_point[i])

    out = tuple(out)
    return out

def datapoint_vis(*args):
    header = [html.Tr([html.Th(i) for i in census_names[:-1]])]

    row = []
    for i, v in enumerate(args):
        if i in map_indices:
            value = map_i_s[map_indices[i]][v]
        else:
            value = v
        row.append(html.Td(value))
    body = [html.Tr(row)]

    return html.Table(
        children=(header + body),
        style={'font-size': '90%', 'width': '100%', 'textAlign': 'center', 'align': 'center'}
    )

def predict(*args):
    pred = clf.predict(np.array([list(args)]))
    pred_mapped = map_i_s['income'][pred[0]]
    return pred_mapped.replace('>', '&gt;')

##############################################################################
########## FATF FUNCTIONALITY
##############################################################################

import fatf.fairness.data.measures as ffdm

# Unfair data rows
def f_d_bias(feature_idx_list):
    data_fairness_mx = ffdm.systemic_bias(data, ground_truth, feature_idx_list)
    true_tuples = np.where(data_fairness_mx == True)
    #
    violating_pairs = []
    for i, j in zip(true_tuples[0], true_tuples[1]):
        pair_a = (i, j)
        pair_b = (j, i)
        if pair_a in violating_pairs or pair_b in violating_pairs:
            pass
        else:
            violating_pairs.append(pair_a)

    header = [html.Tr([html.Th(col) for col in [census_names[-1]]+census_names[:-1]])]

    render_pairs = []
    for pair in violating_pairs:
        render_pairs.append(html.Div(
            children=dcc.Markdown(children='**These are data points with indices {} and {}.**'.format(pair[0], pair[1])),
            style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
        ))
        render_pair = [header[0]]
        for point in pair:
            row = [html.Td(map_i_s['income'][ground_truth[point]])]
            for i, v in enumerate(data[point]):
                if i in map_indices:
                    value = map_i_s[map_indices[i]][v]
                else:
                    value = v
                row.append(html.Td(value))
            render_pair.append(html.Tr(row))
        render_pairs.append(html.Table(
            children=render_pair,
            style={'font-size': '90%', 'width': '100%', 'textAlign': 'center', 'align': 'center'}
        ))

    return render_pairs

import fatf.utils.data.tools as fudt

# Sample size disparity
def f_d_sample(feature_idx_list):
    unique_gt = np.unique(original_ground_truth)
    html_struct = []

    for idx in feature_idx_list:
        feature_name = census_names[idx]

        # Sort out categorical
        if feature_name in map_i_s:
            # is cat
            unique = np.unique(original_data[:, idx])
            splits = [(i, ) for i in unique]
        else:
            splits = None

        html_struct.append(html.H4(
            children='Distribution for feature: {}'.format(feature_name),
            style={'textAlign': 'center', 'color': '#000000'})
        )

        indices_per_bin, bin_names = fudt.group_by_column(original_data, idx, groupings=splits, treat_as_categorical=True)
        counts = [len(i) for i in indices_per_bin]

        if splits is None:
            names = ['{}'.format(i) for i in bin_names]
        else:
            names = [map_i_s[feature_name][i[0]] for i in splits]

        html_struct.append(dcc.Graph(
            id='f-d-counts-{}'.format(feature_name),
            figure={
                'data': [
                    {'x': names, 'y': counts, 'type': 'bar', 'name': feature_name}
                ],
                'layout': {
                    'title': 'Counts per split',
                    'font': {'color': '#000000'}
                }
            })
                           )

        ###
        gt_counts = [[] for i in range(unique_gt.shape[0])]
        for indices_set in indices_per_bin:
            gt_filtered = original_ground_truth[indices_set]
            for i, v in enumerate(unique_gt):
                cnt = (gt_filtered == v).sum()
                gt_counts[i].append(cnt)
        gt_names = [map_i_s['income'][i] for i in unique_gt]

        html_struct.append(dcc.Graph(
            id='f-d-ground-truth-{}'.format(feature_name),
            figure={
                'data': [{'x': names, 'y': count, 'type': 'bar', 'name': name}
                         for count, name in zip(gt_counts, gt_names)],
                'layout': {
                    'title': 'Label per split',
                    'font': {'color': '#000000'}
                }
            })
                           )

    return html_struct
