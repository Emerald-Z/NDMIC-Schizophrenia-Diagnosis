# NDMIC-Schizophrenia-Diagnosis

## Diagnosis of Schizophrenia using Deep Learning and fMRI
### Thesis
Our study aims to enhance schizophrenia diagnosis. Using modern deep learning algorithms applied to resting-state fMRI data, we hypothesize that we can achieve a higher classification accuracy than previous models while extracting clinically meaningful information from the learned features.
### Hypothesis
Current diagnostic methods for schizophrenia are limited by their reliance on subjective assessments and the variable presentation of symptoms. We hypothesize that a deep learning approach based on neurophysiology can achieve a more objective diagnosis. By incorporating high-quality proprietary data into our training and taking known physiological features of the disease into account, we aim to surpass previous classification accuracies while revealing neurophysiological patterns that correlate with disease severity.
### Methodology
We will combine multiple open-source fMRI datasets from schizconnect.org and supplement the data with proprietary fMRI data collected for various studies. We will test multiple methods of preprocessing to determine which method gives the best training data, and then do our own feature engineering. We will subsequently train multiple models on the training data, such as Image Processing Transformers, 3DCNNs, etc, and find the model that performs the best. We will also collaborate with researchers knowledgeable on the physiology of schizophrenia in an attempt to establish a connection between the features the model learned and known markers of the disease.
