# Outlier Detection for NLP using Latent Representations


![GitHub](https://img.shields.io/github/license/jose-melo/nlp-ood-detection) ![Lines of code](https://img.shields.io/tokei/lines/github/jose-melo/nlp-ood-detection)

Welcome to our open-source project on out-of-distribution (OOD) detection with machine learning models! This project aims to compare different methods for OOD detection and provide a benchmark using popular datasets. We also provide open-source code and data to facilitate future experiments in this field.

## Introduction

Machine learning models are known to perform well on data that is close to the train set distribution. However, when they are used on OOD data, they have a high risk of making incorrect predictions, and sometimes they even give high confidence scores to these wrong predictions. This can be a major issue in real-world applications where we need reliable AI. Therefore, it is important to be able to flag OOD cases, so the human user knows that further expertise is required.

Different OOD detection methods have been developed in the last few years, and in this project, we compare some of these methods. Our goals are:

    1. Create a benchmark using the trending methods in OOD detection
    2. Release open-source code and data to ease new experiments

## Folder structure

```bash
.
├── data                              # directory for data
│   ├── ...                           
├── LICENSE                           
├── nlp_ood_detection                 # directory for code related to the natural language processing (NLP) out-of-distribution (OOD) detection
│   ├── aiirw.py                      
│   ├── auroc_and_decoder_methods     # directory for modules related to Area Under the Receiver Operating Characteristics (AUROC) and decoder methods
│   ├── benchmarks                    # directory for modules related to benchmarking
│   │   └── generate_score.py         # Functions that generates the score
│   ├── data_depth                   
│   │   ├── run.py                    # module for running data depth experiments
│   │   ├── similarity_scorer.py      
│   │   └── utils.py                 
│   ├── data_processing              
│   │   ├── bert_datamodule.py        
│   │   ├── data_processing.py        
│   │   └── generate_data.py          # module for generating data
│   ├── models                      
│   │   ├── autoencoder.py            # module for an autoencoder model
│   │   └── bert_based_classifier.py  # module for a BERT-based classifier model
│   ├── run.py                       
│   ├── utils                        
├──  notebooks                      
│   ├── data_depth.ipynb              # notebooks regarding the data depth analysis
│   ├── histograms.ipynb              # notebook with the presented results
│   └── utils.py                      # module containing general utilities for the project

```
## Usage

### Running the OOD detector

You just need to set the parameters:

```python
from nlp_ood_detection.detector import NLPOODDetector

detector_params = {
    "dataset_in": "imdb",
    "dataset_out": "paws",
    "aggregation": "mean",
    "data_folder": "../data",
    "method": "aiirw",
    "model_name": "textattack/distilbert-base-uncased-imdb",
    "max_size": 1000,
    "threshold": 0.9,
}
detector = NLPOODDetector(**detector_params)
```
and then:

```python
detector.fit()
score = detector.score()
prediction = detector.predict()
```

### Generating the data

To generate the data, it is possible to run the following script:

```bash
python nlp_ood_detection/data_processing/generate_data.py --datasets imdb sst2 --aggregations mean last two_last --model_name distilbert-base-uncased --early_stopping 250
```
passing a list of datasets, a list of aggregation methods, the model and a flag of early stopping the propcessing.


### Running the benchmark

You can run the notebook in `notebooks/histograms.ipynb` or run the following
```bash
 python nlp_ood_detection/benchmarks/generate_score.py --datasets imdb ag_news --aggregations mean max --method maha energy
```

## Methodology

We used a distilbert model pre-trained on the imdb dataset to classify text data. We considered sst2 as the input dataset and applied the model to four different test datasets: trec, race, yelp, and paws. We measured the model's performance using the AUC ROC metric. We also used an autoencoder trained on the same imdb dataset to reduce the dimensionality of the data and compared the results with those obtained using the original model.

## Results
Our benchmark results showed that the distilbert model pre-trained on the imdb dataset achieved good results on most datasets, with AUC ROC values ranging from 0.61 to 0.87. When the data was reduced using an autoencoder, the results were comparable to those obtained with the original distilbert model, which could resulted in simplification of the model.

Take a look to the histograms calculated:

![histograms](https://user-images.githubusercontent.com/24592687/233766247-cf81b82a-4186-4916-bf5b-ff5c93cb0810.png)

And here, you can see the corresponding ROC curves:

![roc_curves](https://user-images.githubusercontent.com/24592687/233766248-3ef4126a-a291-4111-873b-16218708d89b.png)
