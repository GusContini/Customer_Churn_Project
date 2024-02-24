'''
Library doc string

Author: GusContini
Date: Feb 2024
'''

import os
import logging
import churn_library as cls
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def test_import(churn_model):
    '''
    Test data import
    '''
    try:
        churn_model.import_data()
        df = churn_model.df
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err
    except pd.errors.EmptyDataError as err:
        logging.error("Testing import_data: The file is empty")
        raise err
    except AssertionError as err:
        logging.error("Testing import_data: Assertion error")
        raise err
    except Exception as e:
        logging.error("Testing import_data: Unexpected error occurred")
        logging.error(str(e))
        raise AssertionError(f"Unexpected error: {str(e)}")


def test_eda(churn_model):
    '''
    Test perform eda function
    '''
    try:
        churn_model.import_data()
        churn_model.perform_eda(
            target='Attrition_Flag',
            target_label='Existing Customer',
            event_id='CLIENTNUM'
        )
        assert os.path.exists('./images/eda/target_histogram.png')
        assert os.path.exists('./images/eda/quant_columns_heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: Assertion error")
        raise err
    except Exception as e:
        logging.error("Testing perform_eda: Unexpected error occurred")
        logging.error(str(e))
        raise AssertionError(f"Unexpected error: {str(e)}")


def test_encoder_helper(churn_model):
    '''
    Test encoder helper
    '''
    try:
        churn_model.import_data()
        churn_model.perform_eda(
            target='Attrition_Flag',
            target_label='Existing Customer',
            event_id='CLIENTNUM'
        )
        churn_model.encoder_helper(
            category_lst=[
                'Education_Level',
                'Income_Category'])
        encoded_df = churn_model.df
        assert 'Education_Level_Churn' in encoded_df.columns
        assert 'Income_Category_Churn' in encoded_df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Assertion error")
        raise err
    except Exception as e:
        logging.error("Testing encoder_helper: Unexpected error occurred")
        logging.error(str(e))
        raise AssertionError(f"Unexpected error: {str(e)}")


def test_perform_feature_engineering(churn_model):
    '''
    Test perform_feature_engineering
    '''
    try:
        churn_model.import_data()
        churn_model.perform_eda(
            target='Attrition_Flag',
            target_label='Existing Customer',
            event_id='CLIENTNUM'
        )
        churn_model.encoder_helper(category_lst=None)
        churn_model.perform_feature_engineering(
            target='Churn', test_fraction=0.3)

        assert churn_model.X_train is not None
        assert churn_model.X_test is not None
        assert churn_model.y_train is not None
        assert churn_model.y_test is not None

        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Assertion error")
        raise err
    except Exception as e:
        logging.error(
            "Testing perform_feature_engineering: Unexpected error occurred")
        logging.error(str(e))
        raise AssertionError(f"Unexpected error: {str(e)}")


def test_train_models(churn_model):
    '''
    Test train_models
    '''
    try:
        churn_model.import_data()
        churn_model.perform_eda(
            target='Attrition_Flag',
            target_label='Existing Customer',
            event_id='CLIENTNUM'
        )
        churn_model.encoder_helper(category_lst=None)
        churn_model.perform_feature_engineering(
            target='Churn', test_fraction=0.3)

        churn_model.train_models(
            param_grid=None,
            clf_1=RandomForestClassifier(random_state=42),
            clf_2=LogisticRegression(solver='lbfgs', max_iter=3000)
        )

        assert churn_model.y_train_preds_clf_1 is not None
        assert churn_model.y_test_preds_clf_1 is not None
        assert churn_model.y_train_preds_clf_2 is not None
        assert churn_model.y_test_preds_clf_2 is not None
        assert churn_model.model_1 is not None
        assert churn_model.model_2 is not None

        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: Assertion error")
        raise err
    except Exception as e:
        logging.error("Testing train_models: Unexpected error occurred")
        logging.error(str(e))
        raise AssertionError(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    # Create an instance of ChurnPredictionModel
    churn_model = cls.ChurnPredictionModel()
    test_import(churn_model)
    test_eda(churn_model)
    test_encoder_helper(churn_model)
    test_perform_feature_engineering(churn_model)
    test_train_models(churn_model)
