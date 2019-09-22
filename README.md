# Deep-Fashion-Attributes

train.py contains training code. 
DataAugmentation.py contains helper code for train.py. Keep them in same folder while executing
test.py contains test code.

Input and Output for each file is same as mentioned in Assignment.

Model is overfitting despite applying regulariation, handling of imbalanced dataset by changing applying weighted loss metric.

Currently model is using few layers of Inception trained on Imagenet as its base model. 
Different base models could have been tested at different hyperparameters for better solution.
