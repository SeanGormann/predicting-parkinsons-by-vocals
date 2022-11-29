# predicting-parkinsons-by-vocals

Here, I'm using Parkinsons Telemonitoring Dataset that contains motor 'Unified Parkinson's Disease Rating Scale' (UPDRS) Score of Parkinsons Disease (PD) patients and controls as well as a number of variables, including a set of variables obtained from vocal recordings. The original paper where the data was obtained can be found here: _Athanasios Tsanas, Max A. Little, Patrick E. McSharry, Lorraine O. Ramig (2009),'Accurate telemonitoring of Parkinson’s disease progression by non-invasive speech tests',IEEE Transactions on Biomedical Engineering._ 
The goal of this project was to build a regression model that would be able to predict the motor UPDRS score based on the set of variables provided. Data was cleaned, models were tested through k-fold cross validation, hyperparameters were tuned and the top scoring model was subsequentely validated on the original test data. 