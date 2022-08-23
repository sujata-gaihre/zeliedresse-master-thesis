# Master Thesis

This repository contains my [thesis](https://github.com/zeliedresse/master-thesis/blob/main/Thesis%20Ze%CC%81lie%20Dresse.pdf), titled *Predicting preterm birth with machine learning methods*, written in order to complete my Msc in Data Analytics and Business Economics at Lund University and the corresponding Python code. The original Natality Birth data files from the National Vital Statistics System are freely available on the [National Bureau of Economic Research website](https://www.nber.org/research/data/vital-statistics-natality-birth-data). This code makes use of the CSV files of 2016-2020 and partially uses the text files for 2020 and 2019.


## Abstract
> Preterm birth is a leading cause for birth complications and neonatal mortality in the world. It remains difficult to predict whether a preterm birth will occur, which hinders the possible use of prevention treatments. This thesis investigates the use of machine learning models in the prediction of spontaneous preterm birth. In addition, possible heterogeneous performance of these models among different racial groups is explored. Using birth certificate data, retrieved from the Natality Birth Data Sets in the National Vital Statistics System, machine learning models were trained and evaluated. Four machine learning methods are employed: logistic regression, random forests, eXtreme gradient boosting and neural networks. The models’ performance is similar across methods, the logistic regression model achieved the lowest test AUC of 0.6710 and the lowest TPR of 30.14% at the 10% FPR level. The eXtreme gradient boosting model performed best with a test AUC of 0.6994 and TPR of 34.15%. All models performed similarly for both black and non-black women. These results confirm previous evidence that this type of easily accessible patient data does not seem to be sufficient to construct high-performing machine learning models.

## Code
The code is uploaded in three different directories:
- Building the dataset: a file for every year to import the data from the csv files and process them in the same way and a file to combine all these years into a final dataset
- Supporting code: a file with helping functions used later on, a file to split, scale and undersample the data and a file to calculate summary statistics
- Models: all the code in which the final models, logistic regression, random forest, eXtreme gradient boosting and neural networks, are built and evaluated
