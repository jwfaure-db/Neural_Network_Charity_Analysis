Neural Network Charity Analysis 
===============================

### Intruduction

The purpose is this project is to use the Neural Network Model to predict success rate for organizations  which received Alphabet Soup’s funding. We will try to build a simple Neural Network Model to evalute the data, and try to improve this model's acurraty using various optimimize techniques.

### Preprocessing Data for a Neural Network Model

The dataset used in this project is charity_data.csv. The dataset
is preprocessed with the following procedures.

The target column is IS_SUCCESSFUL. The 'EIN' and 'NAME' are dropped as these are not indicators of the success.
The rest column are used as features in the initial modeling.

#### Binning
After evaluating, it is found that both the APPLICATION_TYPE and CLASSIFICATION categorical columns have more than 10 unique values. Density plot is used to found cut off values. The lesser frequency values are placed into the "Other" bin for both categories.

#### Encoding
The non-numerical category columns are then encoded using the OneHotEncoder. It is then merged with the original dataframe, with the original non-numerical category columns dropped.

After re-binning and encoding, the data set is divided into training and testing set. Those datasets are then scaled using the StandardScaler.

After these preprocessing, the data sets are ready for modelling.

### Initial Neural Network Model

A simple two-layer neural network model is created for initial model. The first 
hidden layer has 10 neurons and the second layer has 5. The sigmoid activation function 
is used throughout both hidden and output layers. A call back function is used to 
save the weight every 5 epochs.

Below are the evaluation results for this simple model:
```
Training Loss: 0.5495;  Accuracy: 0.7334
Test Loss: 0.560 Accuracy: 0.72396
```
The accuracy of 0.72396 is satisfactory but not stellar. The model results are exported into the *AlphabetSoupCharity.h5* file.

### Optimization

Several attempts are made to improve the accuracy from the initial model with varying degree of success.

#### 1. Dropping Noisy Data Columns, Finer Binning

Results by applying the value_counts function shows the STATUS and SPECIAL_CONSIDERATIONS are heavily skewed,
and they don't have any meaningful value in predicting the success rate. The two columns are dropped from the dataframe.
```
1    34294
0        5
Name: STATUS, dtype: int64
N    34272
Y       27
Name: SPECIAL_CONSIDERATIONS, dtype: int64
```

The initial cut off values for re-binning APPLICATION_TYPE and CLASSIFICATION are set to be 500. Lower
cut off values are being set to have more bins.

Metrics show barely any improvement, which suggest that 
(1) the Neural Network Model is good at filtering out noisy date, and
(2) the initial bining is good enough.

#### 2. Change the activation function

Second method is to replace the *sigmoid* activation function with *relu*, in both the hidden layers
and output layer. As shown below, accuracy barely changed but loss increased.

```
Training loss: 0.6024 - accuracy: 0.7354
Testing Loss: 0.6119405627250671, Accuracy: 0.7238
```

#### 3. Increase the Number of Neurons in Hidden Layers

The third attemp is to increase the neurons in the hidden layer. Here the number of neurons  in the
 first layer has doubled from 10 to 20. Accuracy and loss are now in line with the original model 
 without any meaningful improvement.
```
Training loss: 0.5391 - accuracy: 0.7394
Test Loss: 0.5549247860908508, Accuracy: 0.7244315147399902
```

#### 4. Add Additional Hidden Layers

The fourth method is to add more hidden layer to the neural network. 
A hidden layer with 10 neurons is added, and we increase
the number of first layer nutrons to 20 to handle the added layer. 

There is slight improvement in the model's performance, with testing accuracy improved from 0.724 to 0.726.
```
Training: loss: 0.5363 - accuracy: 0.7403
Test Loss: 0.5544541478157043, Accuracy: 0.7258309125900269
```

More nutrons and additional layers are added to the above network.
The training time increased quickly but the test accuracy are bouncing around
a little but still far from the 0.75 we are aiming for. 
It seems we have reached a limit of this model with our known techniques.

The results are exported into the *AlphabetSoupCharity_Optimization.h5* file.

### Conclusion

A simple two-layer Neural Network Model will predict results with statisfying accurcy. 
The combination of several optimization techniques improved the accuracy slightly, 
but are met with diminishing returns for the additional effort. 
To improved accuracy, it is suggested to explore other machine learning methods
such as Logistic regression, RandomForestClassifier, or Support-vector machine. 
