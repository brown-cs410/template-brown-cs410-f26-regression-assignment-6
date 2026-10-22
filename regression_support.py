"""Provided data, pipeline, and plotting helpers for Assignment 6."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler


def extract_features(df, feature=None):
    """Return numeric features and life expectancy, keeping one row per country."""
    data = df.drop_duplicates(subset=['Country']).select_dtypes(include='number')
    if feature is not None:
        data = data[[feature, 'life_expectancy']]
    data = data.dropna()
    names = data.drop(columns=['life_expectancy']).columns.tolist()
    return (data[names].to_numpy(dtype=float),
            data['life_expectancy'].to_numpy(dtype=float), names)


def prepare_cv_data(df):
    """Remove incomplete rows before choosing one original-scale row per country."""
    return extract_features(df.dropna())


def make_regression_pipeline(regressor, degree=1, feature_indices=None):
    """Fit feature selection and scaling within each training fold."""
    selector = ('passthrough' if feature_indices is None else
                ColumnTransformer([('selected', 'passthrough', list(feature_indices))]))
    return Pipeline([
        ('features', selector),
        ('scale', MinMaxScaler()),
        ('polynomial', PolynomialFeatures(degree=degree, include_bias=False)),
        ('standardize', StandardScaler()),
        ('regressor', regressor),
    ])


def plot_cv_errors(results, parameter, title):
    """Plot fold-mean validation MSE and one standard error for each model."""
    fig, ax = plt.subplots()
    categories = list(dict.fromkeys(results[parameter])) if parameter == 'feature_set' else None
    for name, group in results.groupby('model', sort=False):
        if categories is None:
            group = group.sort_values(parameter)
            x = group[parameter]
        else:
            group = group.set_index(parameter).loc[categories]
            x = np.arange(len(categories))
        ax.errorbar(x, group['mean_validation_mse'],
                    yerr=group['se_validation_mse'], marker='o', capsize=3, label=name, color={'Ridge': 'tab:blue', 'LASSO': 'tab:orange'}[name])
    if parameter == 'beta':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Regularization strength beta')
    elif parameter == 'degree':
        ax.set_xticks(sorted(results['degree'].unique()))
        ax.set_xlabel('Polynomial degree')
    else:
        ax.set_xticks(np.arange(len(categories)), categories)
        ax.set_xlabel('Feature set')
    ax.set_ylabel('Validation MSE (years squared)')
    ax.set_title(title)
    ax.legend()
    fig.savefig(f'figs/cv_{parameter}_error.png', bbox_inches='tight')
    plt.show()
    return ax
