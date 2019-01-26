Readme.txt


Process the original text data file into a Numpy matrix that contains the feature vectors and a Numpy vector that contains the labels.
Partition the data into a training set, a validation set and a testing set. The training set takes around 80% of the total. The validation set takes about 10% . The testing set takes the rest. The three sets should NOT overlap.
For a given group of hyper-parameters such as M, μj, Σj, λ, η(τ), train the model parameter w on the training set.
Validate the regression performance of your model on the validation set. Change your hyper-parameters and repeat step 3. Try to find what values those hyper-parameters should take so as to give better performance on the validation set.
After finishing all the above steps, fix your hyper-parameters and model parameter and test your model’s performance on the testing set. This shows the ultimate effectiveness of your model’s generalization power gained by learning.
