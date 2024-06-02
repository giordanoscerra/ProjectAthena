# This is a general model card for the models we created.
The authors decided that there was no need for multiple model card since all the models have the same objective. And were trained on the same data. The differences are just in the accuracy of such models and in their architecture.

## Models
We developed 3 main model
- Naive Bayes (the lightest model)
- [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) finetuned (a transformed based model with 66 millions parameters)
- [BERT](https://huggingface.co/google-bert/bert-base-uncased) finetuned (a transformer based model with 110 millions parameters)

**input**: text

**output**: one of 13 classes

## Hardware and Software
The training of the transformer based models was done using the HuggingFace _transformer_ library. While the Naive Bayes model was created using _sklearn_ library.

To train the BERT and DistilBERT models we used a _Tesla V100S-PCIE-32GB_ that was provided to us by the university of Pisa. Estimated total emissions for the training and validation of this two model is 3.76 KgCO2eq.

## Training data
The [History of Philosophy](https://www.kaggle.com/datasets/kouroshalizadeh/history-of-philosophy) dataset was split into training (70% of the data), validation (20% of the data) and test (10% of the data).

BERT and DistilBERT were pretrained on [BookCorpus](https://yknzhu.wixsite.com/mbweb) and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia), and have been finetuned by us on the History of Philosophy training set.

Naive Bayes was trained on the History of Philosophy training set, the smoothing hyperparameter has been chosen by comparing validation scores. After model selection, it's been retrained on training+validation set and tested on the test set.
## Results
All the model showed a good generalization capability. We report the results of the f1-score macro average (higher is better) measured on a test set with sentences longer than 83 characters:
- Naive Bayes: 0.81
- DistilBert: 0.87
- Bert: 0.88

## Responsability & Safety

## Ethical Considerations and Limitations


## Contributors

Giordano Scerra, Andrea Marino, Yuri Ermes Negri, Davide Marchi, Davide Borghini
