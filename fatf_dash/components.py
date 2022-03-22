##############################################################################
########## DASHBOARD COMPONENTS
##############################################################################

import dash_core_components as dcc
import dash_html_components as html

import numpy as np

# Data imports
from fatf_dash.census import map_i_s, map_s_i, census_names
from fatf_dash.data_model import data, ground_truth, clf, map_indices


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
            # feature_name = map_indices[i]
            # out.append(map_i_s[feature_name][data_point[i]])
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
