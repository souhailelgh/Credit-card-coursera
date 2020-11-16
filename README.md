* Data Description:
Data Source: \url {https://www.kaggle.com/dalpozz/creditcardfraud/data}
It is a CSV file, contains 31 features, the last feature is used to classify the transaction whether it is a fraud or not.

Information about data set
The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues original features are not provided and more background information about the data is also not present. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
****************************************************************************************************

* The models used are : 

                     *Random Forest and SVM for classfication supervisd machine Learning.
                     * Isolation Forest and Eliptic Envelope for anomaly detection.





Author: ELGHAYAM Souhail engineering student  EHTP CASABLANACA MOROCCO.                  
                     
