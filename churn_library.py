'''
Churn Library doc string: this script creates a class with methods to perform
EDA, feature engineering, train a model to predict churn and save differnt outputs

Author: GusContini
Date: Feb 2024
'''

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class ChurnPredictionModel:
    def __init__(self, data_path='./data/bank_data.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_1 = None
        self.model_2 = None
        self.y_train_preds_clf_1 = None
        self.y_test_preds_clf_1 = None
        self.y_train_preds_clf_2 = None
        self.y_test_preds_clf_2 = None

    def import_data(self):
        '''
        returns dataframe for the csv found at pth

        input:
            pth: a path to the csv
        output:
            df: pandas dataframe
    '''
        self.df = pd.read_csv(self.data_path, index_col=0)

    def perform_eda(
            self,
            target='Attrition_Flag',
            target_label="Existing Customer",
            event_id='CLIENTNUM'):
        # drops 'CLIENTNUM'
        self.df.drop(event_id, axis=1, inplace=True)
        # creates a list of numerical variables
        quant_columns = [
            col for col in self.df.columns if self.df[col].dtype != 'object']
        # creates target variable Churn based on the target parameter
        self.df['Churn'] = self.df[target].apply(
            lambda val: 0 if val == target_label else 1)
        # drops 'Attrition_Flag'
        self.df.drop(target, axis=1, inplace=True)
        # creates a list of categorical variables
        cat_columns = [
            col for col in self.df.columns if self.df[col].dtype == 'object']

        # target distribution
        plt.figure(figsize=(20, 10))
        self.df['Churn'].hist()
        plt.savefig('./images/eda/target_histogram')
        plt.close()

        # numerical univariate plot
        for col in quant_columns:
            plt.figure(figsize=(20, 10))
            sns.histplot(self.df[col], stat='density', kde=True)
            plt.savefig(f'./images/eda/{col}_histogram')
            plt.close()

        # categorical univariate plot
        for col in cat_columns:
            plt.figure(figsize=(20, 10))
            self.df[col].value_counts('normalize').plot(kind='bar')
            plt.savefig(f'./images/eda/{col}_distribution')
            plt.close()

        # numerical variables heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            self.df[quant_columns].corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig(f'./images/eda/quant_columns_heatmap')
        plt.close()

    def encoder_helper(self, category_lst=None):
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
            category_lst = [
                col for col in self.df.columns if self.df[col].dtype == 'object']

        for col in category_lst:
            col_value_groups = self.df.groupby(col)['Churn'].mean()
            col_lst = [col_value_groups.loc[val] for val in self.df[col]]
            self.df.drop(col, axis=1, inplace=True)
            self.df[f'{col}_Churn'] = col_lst

    def perform_feature_engineering(self, target='Churn', test_fraction=0.3):
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
        y = self.df[target]
        self.df.drop(target, axis=1, inplace=True)
        X = self.df.copy()

        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=42)
        
        # Return the values
        return self.X_train, self.X_test, self.y_train, self.y_test

    def classification_report_image(self):
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
        plt.rc('figure', figsize=(8, 5))
        # test set
        plt.text(0.01, 1.25, str('Classifier 1 Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_test, self.y_test_preds_clf_1)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        # train set
        plt.text(0.01, 0.6, str('Classifier 1 Train'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_train, self.y_train_preds_clf_1)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        # save the plot
        plt.savefig('./images/results/clf_1_classification_report')
        plt.clf()

        # Plot clf_2 classification report
        plt.rc('figure', figsize=(8, 5))
        # test set
        plt.text(0.01, 1.25, str('Classifier 1 Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_test, self.y_test_preds_clf_2)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        # train set
        plt.text(0.01, 0.6, str('Classifier 1 Train'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_train, self.y_train_preds_clf_2)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        # save the plot
        plt.savefig('./images/results/clf_2_classification_report')
        plt.clf()

    def feature_importance_plot(self, model, output_pth):
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
        shap_values = explainer.shap_values(self.X_test)
        shap.summary_plot(
            shap_values,
            self.X_test,
            plot_type="bar",
            show=False)
        plt.savefig(output_pth)

        # clf 1 feature importance plot
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [self.X_test.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(15, 18))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(self.X_test.shape[1]), importances[indices])
        plt.xticks(range(self.X_test.shape[1]), names, rotation=90)
        plt.savefig(output_pth)

    def train_models(
        self, param_grid=None, clf_1=RandomForestClassifier(
            random_state=42), clf_2=LogisticRegression(
            solver='lbfgs', max_iter=3000)):
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
        cv_clf_1.fit(self.X_train, self.y_train)

        # classifier 2
        clf_2.fit(self.X_train, self.y_train)

        # predictions of clf_1
        self.y_train_preds_clf_1 = cv_clf_1.best_estimator_.predict(
            self.X_train)
        y_train_preds_clf_1 = self.y_train_preds_clf_1
        
        self.y_test_preds_clf_1 = cv_clf_1.best_estimator_.predict(self.X_test)
        y_test_preds_clf_1 = self.y_test_preds_clf_1

        # predictions of clf_2
        self.y_train_preds_clf_2 = clf_2.predict(self.X_train)
        y_train_preds_clf_2 = self.y_train_preds_clf_2
        self.y_test_preds_clf_2 = clf_2.predict(self.X_test)
        y_test_preds_clf_2 = self.y_test_preds_clf_2

        # model performance plots
        # clf_1
        plt.figure(figsize=(15, 8))
        rfc_plot = RocCurveDisplay.from_estimator(
            cv_clf_1, self.X_test, self.y_test)
        plt.savefig('./images/results/clf_1_roc')

        # clf_2
        plt.figure(figsize=(15, 8))
        lrc_plot = RocCurveDisplay.from_estimator(
            clf_2, self.X_test, self.y_test)
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

        self.model_1 = cv_clf_1.best_estimator_
        self.model_2 = clf_2

        return y_train_preds_clf_1, y_test_preds_clf_1, y_train_preds_clf_2, y_test_preds_clf_2

    def run_pipeline(self):
        '''
        function to execute the whole model process
        '''
        # import dataframe
        self.import_data()
        # perform EDA
        self.perform_eda(
            target='Attrition_Flag',
            target_label="Existing Customer",
            event_id='CLIENTNUM')
        # encode object features
        self.encoder_helper(category_lst=None)
        # perform feature engineering
        self.X_train, self.X_test, self.y_train, self.y_test = self.perform_feature_engineering(
            target='Churn', test_fraction=0.3)
        # train the models
        self.y_train_preds_clf_1, self.y_test_preds_clf_1, self.y_train_preds_clf_2, self.y_test_preds_clf_2 = self.train_models(
            param_grid=None, clf_1=RandomForestClassifier(random_state=42),
            clf_2=LogisticRegression(solver='lbfgs', max_iter=3000))
        # performance report
        self.classification_report_image()
        # feature importance
        self.model_1 = joblib.load('./models/model_1.pkl')
        self.feature_importance_plot(
            model=self.model_1,
            output_pth='./images/results/clf_1_feature_importance')


if __name__ == '__main__':
    churn_model = ChurnPredictionModel()
    churn_model.run_pipeline()
