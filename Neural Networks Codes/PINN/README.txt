The coding files contains the following repository:

1. DNS_data : contains both channel and couette flows (mirrored as well as interpolated data) used by the models
2. model_files : Contains the notebook files for the different models used
3. Save_models: Contains the .pth format file corresponding to the model
4. results : Contains the notebook file where the result using each models can be visualized
5. Result_Comparison: Contains the csv file of the predicted Reynold Stress Tensor got from each model on unseen data
6. Data_preprocessing : notebook file used for preprocessing the data



Since it could take sometime to run the notebook files allowing the creation of the model, the saved model can be load directly and then used on unseen data if needed. 