
import argparse
import os
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from argparse import ArgumentParser
from azureml.core import Workspace, Dataset

subscription_id = '610d6e37-4747-4a20-80eb-3aad70a55f43'
resource_group = 'aml-quickstarts-268633'
workspace_name = 'quick-starts-ws-268633'

workspace = Workspace(subscription_id, resource_group, workspace_name)
dataset = Dataset.get_by_name(workspace, name='divorce-datasets')
# dataset.to_pandas_dataframe()

run = Run.get_context()
x_df = dataset.to_pandas_dataframe().dropna()
y_df = x_df.pop("Class")
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=0)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--solver', type=str, default='lbfgs', help="chose the algorithm to train the model")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Algorithm: ", args.solver)

    # creates the logistic regression model
    model = LogisticRegression(solver=args.solver, C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    # gets and logs the accuracy
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    # dumps the run
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()