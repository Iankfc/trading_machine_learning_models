# trading_machine_learning_models

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)


## About <a name = "about"></a>

**traiding_machine_learning_models** is a Python package that can run multiple pre-determined 
machine learning models. The package allows you to set a multiple set of values for hyper-parameter tuning.


## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [installation](#installation) for notes on how to deploy the project on a live system.

## Prerequisites

First, you need to create a conda virtual environment together with python version 3.9.5 and at the same time install the dependencies in the requirements.txt file.

### Windows CMD Terminal
```
conda create --name my_environment python=3.9.5

```
Next, activate the virtual environment that you just created now. In the windows terminal, type the following commands.

### Windows CMD Terminal
```
conda activate my_environment

```
### Installation

Next, after you have created a conda virtual environment with python version 3.9.5 together with the dependencies in the requirements.txt, you need to pip install sqlconnection (the "Package"). In the windows terminal, type the following codes below.

### Windows CMD Terminal
```
pip install version pip install git+https://github.com/Iankfc/trading_machine_learning_models.git@master
```

If you are having an issue or error while installing the dependencies add the "--no-deps" in the command line to skip the installation of the dependecies.

### Windows CMD Terminal
```
pip install version pip install git+https://github.com/Iankfc/trading_machine_learning_models.git@master --no-deps
```

To use the module in a pythone terminal, import the module just like other python modules such as pandas or numpy.

### Python Terminal
```

from trading_machine_learning_models as etl
import pandas as pd

x, y = make_classification(n_samples=100, random_state=1)
x = pd.DataFrame(x)
y = pd.DataFrame(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                train_size = 0.50,
                                                shuffle= False)
#%% You can manually edit the hyper parameters
nparray_n_estimators = list(np.arange(1,10,1, dtype=int))
nparray_max_depth = list(np.arange(1,10,1, dtype=int))

#%% Use this function to get the dictionary format at which you can manually edit the list of hyper parameters
dict_list_class_ml_classifier_models = etl.func_dict_list_class_ml_classifier_models(RandomForestClassifier = {'n_estimators':nparray_n_estimators,
                                                                                                            'max_depth':nparray_max_depth
                                                                                                            })
print(dict_list_class_ml_classifier_models)

df_prediction = etl.func_df_run_all_models( dict_list_class_ml_classifier_models = dict_list_class_ml_classifier_models,
                                        x_train = x_train,
                                        y_train = y_train,
                                        x_test = x_test,
                                        y_test = y_test)

```


## Usage <a name = "usage"></a>

The module can be use to for extract transform and load (ETL) flow of data science.
