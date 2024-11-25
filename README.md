# IMDb Sentiment Analysis with DistilBERT

## Project Overview

This project uses **DistilBERT**, a lightweight Transformer model, to perform sentiment analysis on IMDb movie reviews. A simple **Gradio interface** is included to demonstrate the model's predictions in real-time.

## Features

- Fine-tuned **DistilBERT** model for sentiment classification.
- **Gradio** interface for an easy-to-use web-based demo.
- IMDb dataset support with customizable training.

## Requirements

To run this project, you need the following dependencies:

- Python 3.8+
- Transformers
- Torch
- Gradio
- Scikit-learn
- Pandas

## Install Dependencies

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```
**Dataset**

The project uses the IMDb dataset for training and testing. Follow these steps to prepare the dataset:

Download the IMDb dataset from [this link](https://ai.stanford.edu/~amaas/data/sentiment/).
Extract the dataset and place it in the 
data/aclImdb/ folder.
The folder structure should look like this:
```
data/aclImdb/

├── train/

│   ├── pos/

│   └── neg/

└── test/
```
**How to Run**



1. Clone this repository:
```
git clone https://github.com/<sh-zohreh>/imdb-analysis.git 
```
2. Navigate into the project directory:
```

cd imdb-analysis
```
3. Install the required libraries:
```
pip install -r requirements.txt
```
4. Ensure the dataset is placed in
```
 data/aclImdb/.
```
5. Run the sentiment analysis application:
```
python src/imdb_analysis.py
```
**License**

This project is licensed under the MIT License. See the LICENSE file for details.

**Contributing**
Feel free to submit issues or pull requests. Contributions are welcome!

**Author**

Zohreh Shafiezadeh Samakoush

