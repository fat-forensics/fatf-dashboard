##############################################################################
########## FATF FUNCTIONALITY
##############################################################################

import dash_core_components as dcc
import dash_html_components as html

import numpy as np

import fatf.fairness.data.measures as ffdm
import fatf.fairness.models.measures as ffmm
import fatf.fairness.predictions.measures as ffpm
import fatf.transparency.predictions.counterfactuals as ftpc
import fatf.utils.data.tools as fudt

# Data imports
from fatf_dash.census import map_i_s, map_s_i, census_names, dtype
from fatf_dash.data_model import original_data, original_ground_truth, original_predictions
from fatf_dash.data_model import data, ground_truth, clf, map_indices


group_fairness_metrics = {'Demographic Parity': 'demographic parity',
                          'Equal Opportunity': 'equal opportunity',
                          'Equal Accuracy': 'equal accuracy'}


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
            treat_as_categorical = True
        else:
            splits = None
            treat_as_categorical = None

        html_struct.append(html.H4(
            children='Distribution for feature: {}'.format(feature_name),
            style={'textAlign': 'center', 'color': '#000000'})
        )


        indices_per_bin, bin_names = fudt.group_by_column(original_data, idx, groupings=splits, treat_as_categorical=treat_as_categorical)
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


def f_m_metrics(features_list, metrics_list):
    html_struct = []
    for f in features_list:
        feature_name = census_names[f]

        # Sort out categorical
        if feature_name in map_i_s:
            # is cat
            splits = [(i, ) for i in np.unique(original_data[:, f])]
            treat_as_categorical = True
        else:
            splits = None
            treat_as_categorical = None

        indices_per_bin, bin_names = fudt.group_by_column(original_data, f, groupings=splits, treat_as_categorical=treat_as_categorical)

        if splits is None:
            names = ['{}'.format(i) for i in bin_names]
        else:
            names = [map_i_s[feature_name][i[0]] for i in splits]

        for m in metrics_list:
            dimp = ffmm.disparate_impact_indexed(
                    indices_per_bin,
                    original_ground_truth,
                    original_predictions,
                    labels=[0, 1],
                    tolerance=0.2,
                    criterion=m)
            dimp_list = dimp.astype(int).tolist()

            html_struct.append(html.H4(
                children='Disparate Impact -- {} -- for feature: {}'.format(m, feature_name),
                style={'textAlign': 'center', 'color': '#000000'})
            )
            html_struct.append(html.Div(
                children=dcc.Graph(
                    id='f-m-heatmap-{}-{}'.format(f, m.replace(' ', '-')),
                    figure={
                        'data': [{
                            'z': dimp_list[::-1],
                            'x': names,
                            'y': names[::-1],
                            'autocolorscale': False,
                            'zmin': 0,
                            'zmax': 1,
                            'zauto': False,
                            # 'colorscale': [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
#                        'text': [
#                            ['a', 'b', 'c'],
#                            ['d', 'e', 'f']
#                        ],
#                        'customdata': [
#                            ['c-a', 'c-b', 'c-c'],
#                            ['c-d', 'c-e', 'c-f'],
#                        ],
                            'type': 'heatmap'
                        }],
                        'layout': {
                            # 'autosize': False,
                            'width': 700,
                            'height': 700

#                    'title': 'Dash Data Visualization',
#                    'font': {
#                        'color': colors['text']
#                    }
                        }
                    }
                ),
                style={'width': '710px', 'textAlign': 'center', 'margin': 'auto'}
            ))




    return html_struct


def f_p_cf(protected_list, datapoint):
    dp = np.array(list(datapoint))
    dpc = clf.predict(dp.reshape(1, -1))[0]
    cidx = [census_names.index(i) for i in map_s_i]
    cidx.remove(14)


    cf, dist, pred = ffpm.counterfactual_fairness(
        instance=dp,
        protected_feature_indices=protected_list,
        model=clf,
        categorical_indices=cidx,
        default_numerical_step_size=10,
        dataset=original_data)

    if cf.size:
        struct_dpc = map_i_s['income'][dpc]

        new_dp = []
        for i, v in enumerate(datapoint):
            feature_name = census_names[i]
            if feature_name in map_i_s:
                new_dp.append(map_i_s[feature_name][v])
            else:
                new_dp.append(v)
        struct_dp = np.array([tuple(new_dp)], dtype=dtype)[0]

        struct_pred = np.array([map_i_s['income'][i] for i in pred])

        struct_cf = []
        for row in cf:
            new_dp = []
            for i, v in enumerate(row):
                feature_name = census_names[i]
                if feature_name in map_i_s:
                    new_dp.append(map_i_s[feature_name][v])
                else:
                    new_dp.append(v)
            struct_cf.append(tuple(new_dp))
        struct_cf = np.array(struct_cf, dtype=dtype)

        ss = ftpc.textualise_counterfactuals(
            struct_dp, struct_cf,
            instance_class=struct_dpc,
            counterfactuals_distances=dist,
            counterfactuals_predictions=struct_pred)
        idd = ss.find('Counterfactual instance') - 0  # 23
        ss = html.Div(
            children=dcc.Markdown(children='```\n{}\n```'.format(ss[idd:])),
                style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
            )
    else:
        ss = html.Div(
                children=dcc.Markdown(children='**No counterfactual instances were found.**'),
                style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
            )

    return ss
