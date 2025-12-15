# Model Card

## Model Details
This model is a supervised binary classification model trained to predict whether an individual earns more than $50K per year based on U.S. Census data. The model was developed as part of an end-to-end machine learning pipeline that includes data preprocessing, training, evaluation, and inference.

Categorical and continuous features are processed using one-hot encoding and normalization. The trained model and encoder are saved to disk for reuse during inference.

## Intended Use
The intended use of this model is educational. It demonstrates how to build, train, evaluate, and deploy a machine learning model using best practices such as modular code, testing, continuous integration, and model documentation.

This model should not be used to make real-world decisions related to employment, compensation, credit, or other high-stakes domains.

## Training Data
The model was trained on the UCI Census Income dataset (census.csv). The dataset contains demographic and employment-related features including age, education, occupation, work class, marital status, and hours worked per week.

The target variable represents whether an individual earns more than $50K per year. The dataset was split into training and test sets prior to model training.

## Evaluation Data
The evaluation data consists of a held-out test split from the same census dataset. This data was not used during training and was used to assess model performance and generalization.

Model performance was also evaluated across categorical feature slices to understand how predictions vary for different subgroups.

## Metrics
The model was evaluated using the following metrics:
- Precision
- Recall
- F1 Score

Model performance on the test dataset was:
- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

Performance metrics for each categorical feature slice were computed and saved to the file `slice_output.txt`.

## Ethical Considerations
The dataset used to train this model may reflect historical and societal biases related to income, occupation, and demographics. As a result, the model may learn and reproduce these biases in its predictions.

This model should not be used in production environments where biased predictions could negatively impact individuals or groups.

## Caveats and Recommendations
- The dataset may not represent current economic conditions.
- Some categorical slices contain very small sample sizes, which can lead to unstable performance metrics.
- The model has not been extensively tuned for optimal performance.

Future work could include bias analysis, hyperparameter tuning, and training on more recent or diverse datasets.
