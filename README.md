# Customer_Churn_Project
## Predict Customer Churn

- Full MLOps Project for **Customer Churn Prediction**

## Project Description
In this project I developed in the churn_python.py file a Class called ChurnPredictionModel.
This classes has several methods:
	- a valid file path containing a csv file must be informed in the class initialization
		data_path='./data/bank_data.csv' is the default parameter
	- import_data(): performs the data importation
	- perform_eda(target='Attrition_Flag', target_label="Existing Customer", event_id='CLIENTNUM'):
		performs EDA and saves images of target distribution, histograms for numerical columns,
		bar plots for categorical columns and a correlation heatmap for numerical columns.
	- encode_helper(category_lst=None): turn each categorical column into a new column with
        propotion of churn for each category
	- perform_feature_engineering(target='Churn', test_fraction=0.3): Divides data into response and
		feature sets and then splits them into train and test sets.
	- train_models(param_grid=None, clf_1=RandomForestClassifier(random_state=42), clf_2=LogisticRegression(
            solver='lbfgs', max_iter=3000)): trains and produces churn predictions and then save the models.
	- classification_report_image(): computes the classification_report method from sklearn.metrics
		and save results as images.
	- feature_importance_plot(model, output_pth): provided a model and a file path, performs feature
		importance and shap model explanation and then save the output plots as images.
There's also the churn_script_logging_and_tests.py where logging and tests are defined.

## Files and data description
	- data folder contains the csv file with the original dataset
	- images folder has two subfolders: eda, containing the outputs of the perform_eda method and results, with
		the model performance outputs.
	- logs folder holds the churn_library.log file, where logs from tests are registered.
	- churn_notebook.ipynb is a notebook used during development
The remaining files are self explanatory.

## Running Files
The main files are churn_library.py and churn_script_logging_and_tests.py and both can be executed in command
prompt using: python script.py (by replacing script.py by either of the scripts).