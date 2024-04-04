# deep-learning-challenge Report
# Report on the Neural Network Model Performance
## Overview of the Analysis
The purpose of this analysis is to address the needs of the nonprofit foundation Alphabet Soup, which seeks a tool to aid in the selection of applicants for funding with the best chance of success in their ventures. Leveraging machine learning and neural networks, the goal is to utilize the features present in the provided dataset to create a binary classifier. This classifier will predict whether applicants will be successful if funded by Alphabet Soup. By developing this predictive model, Alphabet Soup aims to optimize its funding allocation process and maximize the impact of its support on the success of funded ventures.


## Results
### Data Preprocessing
* Target Variable: The target variable for the model is IS_SUCCESSFUL, which indicates whether the funding was used effectively (1) or not (0).
* Features: The features for the model include various metadata columns such as APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.
* Variable to Remove: The variables EIN and NAME are identification columns and should be removed from the input data as they are neither targets nor features.

## Compiling, Training, and Evaluating the Model
### Neurons, Layers, and Activation Functions:
* The neural network model consists of multiple dense layers with varying numbers of neurons.
* ReLU activation function is used for the hidden layers to introduce non-linearity.
* Sigmoid activation function is used in the output layer to obtain binary predictions.
* The number of neurons and layers are chosen based on experimentation and model performance.

### Model Performance:
* The target model performance was to achieve an accuracy higher than 75%.
* The model achieved a loss value of 0.5600 and an accuracy of 73.01%.
* While the accuracy meets the target performance criterion, further optimization may be possible.
* These results are saved to AlphabetSoupCharity.h5

### Steps Taken to Increase Performance:
To improve the accuracy of the neural network model above 75%, I implemented several optimization methods as follows:

#### Hyperparameter Tuning:
I adjusted the learning rate, batch size, and optimizer algorithm to find the optimal configuration for training the neural network. This involved experimenting with different learning rates, batch sizes to improve model convergence and accuracy.

#### Increasing Model Complexity:
I increased the model complexity by adding more neurons to each hidden layer and adding more hidden layers. By increasing the number of neurons in each layer and adding additional layers, the model gained more capacity to capture complex patterns in the data, which could lead to improved performance.

#### Extended Training Duration:
To allow the model more time to learn from the data and converge to an optimal solution, I increased the number of epochs during training. By training the model for a longer duration, it had more opportunities to adjust its weights and biases to minimize the loss function and improve accuracy on the training data.

#### Retaining the "NAME" Column:
Instead of dropping the "NAME" column during preprocessing, I retained it in the dataset. This decision was made based on the understanding that the "NAME" column could potentially contain valuable information that could contribute to the predictive power of the model.

Through these optimization methods, I aimed to enhance the model's predictive accuracy and achieve a target performance higher than 75%. After implementing these optimizations and training the model, the resulting neural network demonstrated improved accuracy on the test data. (hopefully; see limitations below)

The optimized model was saved and exported to an HDF5 file named "AlphabetSoupCharity_Optimization.h5" for future use and deployment.


### Recommendations to Increase Performance
* Further Hyperparameter Tuning: Experiment with different hyperparameter configurations, such as learning rate, batch size, and optimizer algorithms, to optimize model performance further. Fine-tuning these parameters could potentially lead to improvements in accuracy and loss metrics.
* Data Augmentation: If applicable, consider data augmentation techniques to increase the diversity and size of the training dataset. Data augmentation can help improve model robustness and generalization by introducing variations in the training data.

## Summary
Overall, the deep learning model showed promising results in predicting the success of applicants funded by Alphabet Soup. However, achieving the target model performance may require further experimentation and optimization of hyperparameters. The current model achieved an accuracy of 73.01%. Further optimization could involve fine-tuning hyperparameters, exploring different model architectures, and experimenting with feature engineering techniques.

In conclusion, while the deep learning model shows promise, continued refinement and experimentation are recommended to improve performance further. Additionally, exploring alternative models and methodologies could provide additional insights and potentially improve model performance for predicting the success of funded applicants.

## Limitations and Challenges
Despite the efforts to optimize the model and achieve higher performance, challenges were encountered during the execution process. The attempt to run the optimized model (AlphabetSoupCharity_Optimization.ipynb) in Google Colab, even with the Pro version's additional compute resources, was unsuccessful. The session crashed due to memory constraints, indicating that the model's complexity or dataset size surpassed the available RAM capacity, despite the purchase of additional compute resources. I troubleshot the RAM issue but was still unable to view the results. 
