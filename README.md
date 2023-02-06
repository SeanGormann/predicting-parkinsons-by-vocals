# predicting-parkinsons-by-vocals

Here, I'm using Parkinsons Telemonitoring Dataset that contains motor 'Unified Parkinson's Disease Rating Scale' (UPDRS) Score of Parkinsons Disease (PD) patients and controls as well as a number of variables, including a set of variables obtained from vocal recordings. The original paper where the data was obtained can be found here: _Athanasios Tsanas, Max A. Little, Patrick E. McSharry, Lorraine O. Ramig (2009),'Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests',IEEE Transactions on Biomedical Engineering._ 
The goal of this project was to build a regression model that would be able to predict the motor UPDRS score based on the set of variables provided. Data was cleaned, models were tested through k-fold cross validation, hyperparameters were tuned and the top scoring model was subsequentely validated on the original test data. 

I began by carrying out some exploratory data analysis. A long with a description of the data & a correlation matrix, I also made some histogram charts to get a feel for the data:

![image](https://user-images.githubusercontent.com/100109163/216932390-93d41a68-4eda-44a1-9faf-86a6a1e1f953.png)


After carrying out all the necessary data cleaning steps involved, it was time to train the model. The data was split into training and testing cohorts. Numerour models were trained on the training data. Further K fold cross validation was used and regression statistics were generated to determine which model was performing the best. Random forest regressor was determinted to be the most optimal and after some hyperparameter tuning, it was tested on the test data set. Here is a graph of the results: 

![image](https://user-images.githubusercontent.com/100109163/216931770-d1405983-0b4e-4253-bee8-01d1a449c6bb.png)
