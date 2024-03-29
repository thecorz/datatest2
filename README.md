README.md

# Data Scientist: work test (brief)
Please work your way through the challenge below. This is not designed to trip you up and you may refer to any resources necessary.
* Resources: your own know-how plus feel free to use any resource you can find online, this is not a memory test.
* Not allowed: sharing...or help from a friend, colleague, relative or basically anyone else!
* Time allowed: 60 minutes. Your time is precious.
* Assessment: inspection of method used, expected results obtained. 
* Completed? Please export as a Python module and return to the recruiter.

## Challenge
Build a simple model using the nominated dataset and Sklearn (or other Python library) to predict __"Level"__ based on any available and appropriate features.
#### Inputs
Use this dataset: http://resources.xperthr.co.uk/surveys/salary/Sample/Work_Test_-_synthetic_data_ds.xlsx
#### Outputs
* Working code that fits a model 
* Prediction accuracy score metric named "c1_score" for a 30% test sample


## SOLUTION

### Set up
Install the packages in `requirements.txt` file. Run

`git install requirements.txt`

### EDA
EDA in jupyter notebook `code/data_scientist_work_test_brief.ipynb`

### Model module
module in `xperthr_data_test.py`

the workflow is in `run.py`. Run the script with the file `data/Work_Test_-_synthetic_data_ds.xlsx` to get the 'f1_score' of the model:

`python3 run.py data/Work_Test_-_synthetic_data_ds.xlsx`