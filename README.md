[![Render](https://img.shields.io/badge/view-dashboard-blue)](https://fatf.onrender.com/)
[![new BSD](https://img.shields.io/github/license/fat-forensics/fatf-dashboard.svg)](https://github.com/fat-forensics/fatf-dashboard/blob/master/LICENCE)

# FAT Forensics Dashboard Example #

> :warning: The dashboard has been migrated from [Heroku](https://fatf.herokuapp.com/) to
> [Render](https://fatf.onrender.com/) due to the former platform abandoning the free tier.

This repository holds the source code for a simple FAT Forensics dashboard built with [dash](https://dash.plotly.com/).
You can preview the deployment at <https://fatf.onrender.com/>; please allow 30 seconds for the dashboard to load.
This example is based on a *logistic regression* model trained with scikit-learn on the [Adult](https://archive.ics.uci.edu/ml/datasets/adult) data set.
For more information see the
*FAT Forensics: A Python Toolbox for Algorithmic Fairness, Accountability and Transparency*
paper [published in the *Software Impacts* journal](https://www.sciencedirect.com/science/article/pii/S2665963822000951).

## Preparing Data and Models ##

A pre-processed data set (together with the labels) and a trained model are included in the `_data_model` directory.
These can be regenerated with the `prepare_data_model.ipynb` Jupyter Notebook included therein.

## Local Deployment ##

The dashboard can be deployed locally by installing the dependencies
``` bash
pip install -r requirements.txt
```
launching `gunicorn`
``` bash
gunicorn --workers=2 app:server
```
and navigating to the local deployment accessible at <http://127.0.0.1:8000/>.
