*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

This project is the final capstone project of the Udacity Azure ML Nanodegree. In this project, two models are created: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. The performance of the two models is then compared and the best performing model deployed. Finally the endpoint produced will be used to get some answers about predictions.
![alt text](images/capstone-diagram.png)

## Project special Installation
In this repo I created the following files, which required to run the experiments:

- **automl.ipynb** : It is the notebook file for the AutoML (endpoint python script I written in here used to consume the produced endpoint)
- **train.py** : A python script that the HyperDrive operates on in order to produce the runs and find the best model.
- **hyperparameter_tuning.ipynb** : This is the notebook file I used for the HyperDrive.
- **Divorce-Predictor-Dataset.csv** : This is the dataset I used from here. The following came out from the running of the experiments
- **conda_dependencies.yml** : This is the environment file I downloaded from the Azure ML Studio.
- **hyper-model.pkl** : This is the best model from the HyperDrive I downloaded from Azure ML studio.
- **model.pkl** : This is the best model from the AutoML I downloaded from Azure ML studio.
- **hyper_scoring.py** : This is the scrore result from the HyperDrive training
- **automl_scoring.py** : This is the scrore result from the AutoML training

## Dataset

### Overview
The dataset I used is [Divorce-Predictor-Dataset.csv](https://www.kaggle.com/datasets/rabieelkharoua/split-or-stay-divorce-predictor-dataset/data) from Kaggle. it's The Divorce Predictors Scale (DPS) dataset which is derived from a study focused on predicting divorce using the DPS within the framework of Gottman couples therapy. The dataset comprises responses collected from participants, consisting of both divorced and married couples.
Attributes:

### The dataset features the following attributes ### 

- Participant ID: Unique identifier for each participant.
- Marital Status: Indicates whether the participant is divorced or married.
- Demographic Information: Includes age, gender, education level, and other relevant demographic factors.
- Responses to DPS Items: Each item of the DPS is represented as a separate attribute, providing insight into the participants' perceptions and behaviors related to marital dynamics.

-There are 54 questions, labeled Atr1 to Atr54, corresponding to Question 1 to Question 54.

-The last column is the status column, which indicates whether the individual is 'Married' or 'Divorced'. It is represented by a Boolean variable, where 'Married' is represented as '1' and 'Divorced' as '0'."

### Data Statistics ###
![alt text](<images/dataset - distribution.png>)

### Task
the dataset released by the researchers includes ONLY the questions, responses, and the marital status (married or divorced).

[Survey questions](https://www.kaggle.com/datasets/rabieelkharoua/split-or-stay-divorce-predictor-dataset/data) can see in here from kanggle 

### Access

- The Azure Auto ML notebook reads the data using Dataset.Tabular.from_delimeted_files() and registers is as an Azure tabular dataset in the workspace.

- For the hyperparameter tuning, the data is loaded into the workspace using TabularDataFactory in the train.py script.

## Automated ML
## Automated ML configuration ##
Overview of the AutoML settings and configuration used for this experiment:

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
## Defining a Model Training

I use Logistric Regression algorithm from the SKLearn framework in conjuction with hyperDrive for hyperparameter tuning. There are two hyperparamters for this experiment, C and max_iter. C is the inverse regularization strength whereas max_iter is the maximum iteration to converge for the SKLearn Logistic Regression.

## Defining the Hyperparamters tuning

I use Logistric Regression algorithm from the SKLearn framework in conjuction with hyperDrive for hyperparameter tuning. There are two hyperparamters for this experiment, C and max_iter. 
- **C** is the inverse regularization strength 
- **max_iter** is the maximum iteration to converge for the SKLearn Logistic Regression.

We have used random parameter sampling to sample over a discrete set of values. Random parameter sampling is great for discovery and getting hyperparameter combinations that you would not have guessed intuitively, although it often requires more time to execute.
We have used random parameter sampling to sample over a discrete set of values. Random parameter sampling is great for discovery and getting hyperparameter combinations that you would not have guessed intuitively, although it often requires more time to execute.



After that, I will define an early termnination policy. The BanditPolicy basically states to check the job every 1 iterations. If the primary metric (defined later) falls outside of the top 20% range, Azure ML terminate the job. This saves us from continuing to explore hyperparameters that don't show promise of helping reach our target metric.

early_termination_policy = BanditPolicy(evaluation_interval=1, slack_factor=0.2)

I'm ready to configure a run configuration object, and specify the primary metric validation_acc that's recorded in your training runs. I also set the number of samples to 50, and maximal concurrent job to 4, which is the same as the number of nodes in our computer cluster.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
