##############################################################################
########## DASHBOARD APPLICATION
##############################################################################

import fatf
import dash

import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State

# Data imports
from fatf_dash.census import census_names
# Dashboard components and functions
from fatf_dash.components import (
    generate_table, datapoint_selection, random_point, datapoint_vis, predict)
from fatf_dash.fatf import (
    group_fairness_metrics, f_d_bias, f_d_sample, f_m_metrics, f_p_cf)


DATAPOINT = None


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True
# app.config.suppress_callback_exceptions = True
server = app.server
colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'green': '#B8BB26',
    'black': '#000000'
}


app.layout = html.Div(
    style={
        # 'backgroundColor': colors['background']
        'width': '90%',
        'margin': 'auto'
    },

    children=[
        html.H1(
            children='FAT Forensics ({}) Deployment Dashboard Preview'.format(fatf.__version__),
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        html.H3(
            children=(
                'FAT-Forensics:\n'
                'A Python Toolbox for Algorithmic Fairness, Accountability and Transparency'),
            style={
                'textAlign': 'center',
                'color': colors['green']
            }
        ),

        html.H3(
            children=[
                '(',
                html.A(children='arXiv:1909.05167',
                       href='https://arxiv.org/abs/1909.05167'),
                 ')'],
            style={
                'textAlign': 'center',
                'color': colors['green']
            }
        ),

        html.H4(
            children='Kacper Sokol, Raul Santos-Rodriguez and Peter Flach',
            style={
                'textAlign': 'center',
                'color': colors['green']
            }
        ),

        html.H4(
            children=[
                '(The source code of this dashboard is available on ',
                html.A(children='GitHub',
                       href='https://github.com/fat-forensics/fatf-dashboard/'),
                '.)'],
            style={
                'textAlign': 'center',
                'color': colors['black']
            }
        ),

        html.Div(
            children=dcc.Markdown(children=(
                '---\n\n'
                'This dashboard showcases various Fairness, Accountability and Transparency (FAT) '
                'functionality implemented by the FAT Forensics Python package. '
                'It allows the user to inspect FAT of data, models and predictions for a '
                'scikit-learn logistic regression model trained on the Adult (US Census) data set.'
                '\n\n---')
            ),
            style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
        ),

        # Data table
        html.H3(children='Adult Data Set (Random Sample)',
                style={'textAlign': 'center' }
                ),
        generate_table(5),

        html.H3(children='Select a data point for inspection; remember to click the Submit button',
                style={'textAlign': 'center' }
                ),
        datapoint_selection(),
        html.Button('Submit', id='submit_button', n_clicks=0),
        html.Button('Lucky Dip', id='lucky_dip_button', n_clicks=0),

        html.Div(
            children=dcc.Markdown(children='---'),
            style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
        ),
        html.Div(
            children='',
            id='datapoint_text'
        ),
        html.Div(
            children=dcc.Markdown(children='---'),
            style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
        ),

        html.H3(children='Inspection Panel',
                style={'textAlign': 'center'}
                ),

        # tabs: FAT
        html.Div(
            children=[
                dcc.Tabs(
                    id='fat-tabs',
                    style={'height': '20', 'verticalAlign': 'middle'},
                    children=[
                        dcc.Tab(label='Fairness', value='fairness'),
                        dcc.Tab(label='Accountability', value='accountability'),
                        dcc.Tab(label='Transparency', value='transparency'),
                    ],
                    value='fairness',
                )],
            className='row tabs_fat'
        ),

        # tabs: data, models, predictions
        html.Div(
            children=[
                dcc.Tabs(
                    id='dmp-tabs',
                    style={'height':'20', 'verticalAlign': 'middle'},
                    children=[
                        dcc.Tab(label='Data', value='data'),
                        dcc.Tab(label='Models', value='models'),
                        dcc.Tab(label='Predictions', value='predictions'),
                    ],
                    value='data',
                )],
            className='row tabs_dmp'
        ),

        # Tab content
        html.Div(
            children='',
            id='tabs-content'
        )

    ]
)


fairness_data = [
    html.H4(
        children='Please select protected attributes (e.g., *sex* and/or *race*):',
        style={
            'textAlign': 'center',
            'color': colors['black']
        }
    ),

    dcc.Dropdown(
        id='f-d-protected',
        options=[{'label': v, 'value': i} for i, v in enumerate(census_names[:-1])],
        value='',
        placeholder='Select protected attributes...',
        multi=True
    ),

    html.Button('Submit', id='f-d-submit_button', n_clicks=0),

    html.Div(
        children=dcc.Markdown(children=(
            'The collection of tables displayed below will capture a pair of data '
            'points that differ in both protected features and label, i.e. unfair rows in '
            'the training data set. '
            'The bar plots will show the number of instances for each unique value of '
            'the protected feature and the label distribution therein.')),
        style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
    ),

    html.Div(
        id='f-d-out',
        children='Fairness/Data is under construction.'
    )
]


fairness_models = [
    html.H4(
        children='Please select protected attributes:',
        style={
            'textAlign': 'center',
            'color': colors['black']
        }
    ),

    dcc.Dropdown(
        id='f-m-protected',
        options=[{'label': v, 'value': i} for i, v in enumerate(census_names[:-1])],
        value='',
        placeholder='Select protected attributes...',
        multi=True
    ),

    html.H4(
        children='Please select metrics:',
        style={
            'textAlign': 'center',
            'color': colors['black']
        }
    ),

    dcc.Dropdown(
        id='f-m-metrics',
        options=[{'label': d, 'value': v} for d, v in group_fairness_metrics.items()],
        value='',
        placeholder='Select protected attributes...',
        multi=True
    ),

    html.Button('Submit', id='f-m-submit_button', n_clicks=0),

    html.Div(
        children=dcc.Markdown(children=(
            'The heatmaps show the selected disparate impact metric for each pair of '
            'possible values for the selected protected feature.\n\n'
            '**0** (blue) indicates no disparate impact. '
            '**1** (red) indicates disparate impact.')),
        style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
    ),

    html.Div(
        id='f-m-out',
        children=['Fairness/Models is under construction.']
    )
]


fairness_predictions = [
    html.H4(
        children='Please select protected attributes:',
        style={
            'textAlign': 'center',
            'color': colors['black']
        }
    ),

    dcc.Dropdown(
        id='f-p-protected',
        options=[{'label': v, 'value': i} for i, v in enumerate(census_names[:-1])],
        value='',
        placeholder='Select protected attributes...',
        multi=True
    ),

    html.Button('Submit', id='f-p-submit_button', n_clicks=0),

    html.Div(
        children=dcc.Markdown(children=(
            'Counterfactual fairness:')),
        style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
    ),

    html.Div(
        id='f-p-out',
        children=['Fairness/Models is under construction.']
    )
]


accountability_data = [
    'Accountability/Data is under construction.',

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': ['a', 'b', 'c'], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': ['a', 'b', 'c'], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization',
                # 'plot_bgcolor': colors['background'],
                # 'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
]


accountability_models= 'Accountability/Models is under construction.'
accountability_predictions = 'Accountability/Predictions is under construction.'


transparency_data = 'Transparency/Data is under construction.'
transparency_models= 'Transparency/Models is under construction.'
transparency_predictions = 'Transparency/Predictions is under construction.'


# Input fields +  button functionality
@app.callback(
    Output('datapoint_text', 'children'),
    [Input('submit_button', 'n_clicks')],
    [State('{}_input'.format(name), 'value') for name in census_names[:-1]]
)
def update_graph(n_clicks, *args):
    # age, workclass, fnlwgt, education, education_num, marital_status,
    # occupation, relationship, race, sex, capital_gain, capital_loss,
    # hours_per_week, native_country
    global DATAPOINT
    if n_clicks < 1:
        prediction = 'Please declare the data point to get the prediction.'
        dpv = ''
    else:
        is_complete = True
        feature_name = 'UNKNOWN'
        for i, v in enumerate(args):
            if v == '' or v is None:
                feature_name = census_names[i]
                is_complete = False
                break

        if is_complete:
            DATAPOINT = args
            prediction = predict(*args)
            dpv = datapoint_vis(*args)
        else:
            prediction = 'The data point is ill-specified. Cannot predict. Incorrect \_\_{}\_\_ feature. ({})'.format(feature_name, n_clicks)
            dpv = ''
            DATAPOINT = None

    prediction_vis = html.Div(
        children=dcc.Markdown(children='###### Prediction: ######\n**{}**'.format(prediction)),
        style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
    )

    return [dpv, prediction_vis]


# Lucky dip button
@app.callback(
    [Output('{}_input'.format(name), 'value') for name in census_names[:-1]],
    [Input('lucky_dip_button', 'n_clicks')]
)
def lucky_dip(n_clicks):
    return random_point()

@app.callback(
    Output('tabs-content', 'children'),
    [Input('fat-tabs', 'value'), Input('dmp-tabs', 'value')]
)
def render_content(fat_tab, dmp_tab):
    if fat_tab == 'fairness':
        if dmp_tab == 'data':
            return fairness_data
        elif dmp_tab == 'models':
            return fairness_models
        elif dmp_tab == 'predictions':
            return fairness_predictions
        else:
            return 'UNKNOWN FAIRNESS DMP TAB.'
    elif fat_tab == 'accountability':
        if dmp_tab == 'data':
            return accountability_data
        elif dmp_tab == 'models':
            return accountability_models
        elif dmp_tab == 'predictions':
            return accountability_predictions
        else:
            return 'UNKNOWN ACCOUNTABILITY DMP TAB.'
    elif fat_tab == 'transparency':
        if dmp_tab == 'data':
            return transparency_data
        elif dmp_tab == 'models':
            return transparency_models
        elif dmp_tab == 'predictions':
            return transparency_predictions
        else:
            return 'UNKNOWN TRANSPARENCY DMP TAB.'
    else:
        return 'UNKNOWN FAT TAB.'


@app.callback(
    Output('f-d-out', 'children'),
    [Input('f-d-submit_button', 'n_clicks')],
    [State('f-d-protected', 'value')]
)
def f_d_protected(n_clicks, protected_list):
    if n_clicks < 1:
        return html.Div(
            children=dcc.Markdown(children='**Please select protected features to see the results.** (E.g., *sex* and/or *race*.)'),
            style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
        )
    else:
        if not protected_list or protected_list is None:
            return html.Div(
                children=dcc.Markdown(children='**No protected features were selected.**'),
                style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
            )
        else:
            vld = f_d_bias(protected_list)
            fgs = f_d_sample(protected_list)
            return vld + fgs


@app.callback(
    Output('f-m-out', 'children'),
    [Input('f-m-submit_button', 'n_clicks')],
    [State('f-m-protected', 'value'), State('f-m-metrics', 'value')]
)
def f_m_protected(n_clicks, protected_list, metrics_list):
    if n_clicks < 1:
        return html.Div(
            children=dcc.Markdown(children='**Please select protected features to see the results.** (E.g., *sex* and/or *race*.)'),
            style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
        )
    else:
        if not protected_list or not metrics_list or protected_list is None or metrics_list is None:
            return html.Div(
                children=dcc.Markdown(children='**No protected features or metrics were selected.**'),
                style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
            )
        else:
            return f_m_metrics(protected_list, metrics_list)


@app.callback(
    Output('f-p-out', 'children'),
    [Input('f-p-submit_button', 'n_clicks')],
    [State('f-p-protected', 'value')]
)
def f_p_protected(n_clicks, protected_list):
    if n_clicks < 1:
        return html.Div(
            children=dcc.Markdown(children='**Please select protected features to see the results.** (E.g., *sex* and/or *race*.)'),
            style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
        )
    else:
        if DATAPOINT is None:
            return html.Div(
                children=dcc.Markdown(children='**A data point was not submitted.**'),
                style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
            )
        elif not protected_list or protected_list is None:
            return html.Div(
                children=dcc.Markdown(children='**No protected features were selected.**'),
                style={'width': '80%', 'textAlign': 'center', 'margin': 'auto'}
            )
        else:
            return f_p_cf(protected_list, DATAPOINT)


if __name__ == '__main__':
    app.run_server(debug=True)
