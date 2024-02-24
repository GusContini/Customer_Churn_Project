'''
Library doc string
Author: GusContini
Date: Feb 2024
'''

# import libraries
import os
import joblib
import shap
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth, index_col=0)
    return df


def perform_eda(
        df,
        target='Attrition_Flag',
        target_label="Existing Customer",
        event_id='CLIENTNUM'):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            target: y variable containing the model target
            target_label: value indicating True to the target variable
            event_id: variable that contains the unique identifier for the event represented
                in each entry of the dataset

    output:
            None
    '''
    # drops 'CLIENTNUM'
    df.drop(event_id, axis=1, inplace=True)
    # creates a list of numerical variables
    quant_columns = [col for col in df.columns if df[col].dtype != 'object']
    # creates target variable Churn based on the target parameter
    df['Churn'] = df[target].apply(lambda val: 0 if val == target_label else 1)
    # drops 'Attrition_Flag'
    df.drop(target, axis=1, inplace=True)
    # creates a list of categorical variables
    cat_columns = [col for col in df.columns if df[col].dtype == 'object']

    # target distribution
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('./images/eda/target_histogram')

    # numerical univariate plot
    for col in quant_columns:
        plt.figure(figsize=(20, 10))
        sns.histplot(df[col], stat='density', kde=True)
        plt.savefig(f'./images/eda/{col}_histogram')

    # categorical univariate plot
    for col in cat_columns:
        plt.figure(figsize=(20, 10))
        df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig(f'./images/eda/{col}_distribution')

    # numerical variables heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df[quant_columns].corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(f'./images/eda/quant_columns_heatmap')

    pass


def encoder_helper(df, category_lst=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            target: string of response name [optional argument that could be used
                for naming variables or index y column]
            target_label: value indicating True to the target variable

    output:
            df: pandas dataframe with new columns
    '''
    # creates a list of categorical variables if None is provided
    if category_lst is None:
        category_lst = [col for col in df.columns if df[col].dtype == 'object']

    for col in category_lst:
        col_value_groups = df.groupby(col)['Churn'].mean()
        col_lst = [col_value_groups.loc[val] for val in df[col]]
        df.drop(col, axis=1, inplace=True)
        df[f'{col}_Churn'] = col_lst

    return df


def perform_feature_engineering(df, target='Churn', test_fraction=0.3):
    '''
    input:
              df: pandas dataframe
              target: string of response name [optional argument that could be
                used for naming variables or index y column]
              test_fraction: proportion saved for the validation set (0 < test_fraction < 1)

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df[target]
    df.drop(target, axis=1, inplace=True)
    X = df.copy()

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_clf_2,
                                y_train_preds_clf_1,
                                y_test_preds_clf_2,
                                y_test_preds_clf_1):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Plot clf_1 classification report
    plt.clf()
    plt.rc('figure', figsize=(8, 5))
    # test set
    plt.text(0.01, 1.25, str('Classifier 1 Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_clf_1)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    # train set
    plt.text(0.01, 0.6, str('Classifier 1 Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_clf_1)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    # save the plot
    plt.savefig('./images/results/clf_1_classification_report')

    # Plot clf_2 classification report
    plt.clf()
    plt.rc('figure', figsize=(8, 5))
    # test set
    plt.text(0.01, 1.25, str('Classifier 1 Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_clf_2)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    # train set
    plt.text(0.01, 0.6, str('Classifier 1 Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_clf_2)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    # save the plot
    plt.savefig('./images/results/clf_2_classification_report')

    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # clf 1 model explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(output_pth)

    # clf 1 feature importance plot
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(15, 18))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)

    pass


def train_models(X_train, X_test, y_train, y_test, param_grid=None,
                 clf_1=RandomForestClassifier(random_state=42),
                 clf_2=LogisticRegression(solver='lbfgs', max_iter=3000)):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              param_grid: grid of hyperparameters to be optimized
              clf_1: Classifier 1 (will have hyperparameters tuned)
              clf_2: Classifier 2
    output:
              y_train_preds_clf_1: clf_1 predictions for train set
              y_test_preds_clf_1: clf_1 predictions for test set
              y_train_preds_clf_2: clf_2 predictions for train set
              y_test_preds_clf_2: clf_2 predictions for test set
    '''
    # grid search
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    if param_grid is None:
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

    # classifier 1
    cv_clf_1 = GridSearchCV(estimator=clf_1, param_grid=param_grid, cv=5)
    cv_clf_1.fit(X_train, y_train)

    # classifier 2
    clf_2.fit(X_train, y_train)

    # predictions of clf_1
    y_train_preds_clf_1 = cv_clf_1.best_estimator_.predict(X_train)
    y_test_preds_clf_1 = cv_clf_1.best_estimator_.predict(X_test)

    # predictions of clf_2
    y_train_preds_clf_2 = clf_2.predict(X_train)
    y_test_preds_clf_2 = clf_2.predict(X_test)

    # model performance plots
    # clf_1
    plt.figure(figsize=(15, 8))
    rfc_plot = RocCurveDisplay.from_estimator(cv_clf_1, X_test, y_test)
    plt.savefig('./images/results/clf_1_roc')

    # clf_2
    plt.figure(figsize=(15, 8))
    lrc_plot = RocCurveDisplay.from_estimator(clf_2, X_test, y_test)
    plt.savefig('./images/results/clf_2_roc')

    # clf_1 and clf_2
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_plot.plot(ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/clf_1_clf_2_roc')

    # save best model
    joblib.dump(cv_clf_1.best_estimator_, './models/model_1.pkl')
    joblib.dump(clf_2, './models/model_2.pkl')

    return y_train_preds_clf_1, y_test_preds_clf_1, y_train_preds_clf_2, y_test_preds_clf_2


if __name__ == '__main__':
    # import dataframe
    df = import_data('./data/bank_data.csv')
    # perform EDA
    perform_eda(
        df,
        target='Attrition_Flag',
        target_label="Existing Customer",
        event_id='CLIENTNUM')
    # encode object features
    df = encoder_helper(df, category_lst=None)
    # perform feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, target='Churn', test_fraction=0.3)
    # train the models
    y_train_preds_clf_1, y_test_preds_clf_1, y_train_preds_clf_2, y_test_preds_clf_2 = train_models(
        X_train, X_test, y_train, y_test, param_grid=None, clf_1=RandomForestClassifier(
            random_state=42), clf_2=LogisticRegression(
            solver='lbfgs', max_iter=3000))
    # performance report
    classification_report_image(y_train, y_test,
                                y_train_preds_clf_2, y_train_preds_clf_1,
                                y_test_preds_clf_2, y_test_preds_clf_1)
    # feature importance
    model = joblib.load('./models/model_1.pkl')
    feature_importance_plot(model, X_test, './images/results/clf_1_feature_importance')
