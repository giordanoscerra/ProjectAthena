# This is a general model card for the models we created.
The authors decided that there was no need for multiple model card since all the models have the same objective. And were trained on the same data. The differences are just in the accuracy of such models and in their architecture.

## Models
We developed 3 main model
- Naive Bayes (the lightest model)
- DistilBERT finetuned (a transformed based model with 66 millions parameters)
- BERT finetuned (a transformer based model with 110 millions parameters)

**input**: text

**output**: one of 13 classes

## Hardware and Software
The training of the transformer based models was done using the HuggingFace transformer library. While the Naive Bayes model was created using 
## Training data

## Results

## Responsability & Safety

## Ethical Considerations and Limitations


## Contributors

Giordano Scerra, Andrea Marino, Yuri Ermes Negri, Davide Marchi, Davide Borghini