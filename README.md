Unified-Adversarial-Invariance
===========

This repository provides access to code for the papers:

1. &nbsp;   &nbsp;   _A. Jaiswal, Y. Wu, W. AbdAlmageed, and P. Natarajan, "Unsupervised Adversarial Invariance" (NeurIPS, 2018)_

2. &nbsp;   &nbsp;  _A. Jaiswal, Y. Wu, W. AbdAlmageed, and P. Natarajan, "Unified Adversarial Invariance" (arXiv, 2019)_.

The first paper presents an approach for invariance to nuisance factors of data through learning a split representation of data and the second extends the approach to additionally induce invariance to biasing factors of data.


## Dependencies

The code is written in Python 2.7 and has the following dependencies.

| Package | Version | Source |
| -------- | :--------: | :--------: |
| joblib | 0.11 | pip |
| NumPy | 1.14.0 | pip |
| SciPy | 1.0.0 | pip |
| TensorFlow | 1.8.0 | pip |
| Keras | 2.1.2 | pip |
| keras-adversarial | --- | [GitHub](https://github.com/bstriner/keras-adversarial) |

The code has been tested with the versions of these dependencies as specified above.


## Usage

1. Install dependencies listed above

2. Clone this repository

3. Update the PYTHONPATH environment variable to include this repository. For Linux/MacOS users, the following can be added to the `~/.bashrc` file:
    ```bash
    export PYTHONPATH=/path/to/Unified-Adversarial-Invariance:$PYTHONPATH
    ```
4. Training examples

    <ins>Nuisance</ins>:
    ```bash
    python train.py mnist_rot mnist_rot_model \
                    /path/to/weights/root/ \
                    --bias 0 --streaming-data 1 \
                    --predictor-loss-weight 100 \
                    --decoder-loss-weight 0.1 \
                    --disentangler-loss-weight 1 \
                    --epochs 10000
    ```
    <ins>Bias</ins>:
    ```bash
    python train.py german german_model_1 \
                    /path/to/weights/root/ \
                    --bias 1 --fold-id 1 --streaming-data 0 \
                    --predictor-loss-weight 100 \
                    --decoder-loss-weight 0.1 \
                    --disentangler-loss-weight 1 \
                    --z-discriminator-loss-weight 1 \
                    --epochs 10000
    ```
    
   For more details and options:
   ```bash
   python train.py -h
   ```
6. Prediction examples

    ```bash
    python predict.py mnist_rot mnist_rot_model /path/to/weights/root/ 9999 \
                      test test_mnist_rot.npy --streaming-data 1
    ```
    ```bash
    python predict.py german german_model /path/to/weights/root/ 9999 \
                      test test_german_1.npy --streaming-data 0 --fold-id 1
    ```
   For more details and options:
   ```bash
   python predict.py -h
   ```

## Citation

Please cite **both** of our following papers with the BibTeX:

``` latex
@incollection{jaiswal2018uai,
    title = {{Unsupervised Adversarial Invariance}},
    author = {Jaiswal, Ayush and Wu, Rex Yue and Abd-Almageed, Wael and Natarajan, Prem},
    booktitle = {Advances in Neural Information Processing Systems 31},
    editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
    pages = {5097--5107},
    year = {2018},
    publisher = {Curran Associates, Inc.}
} 

@article{jaiswal2019unifai,
    title = {{Unified Adversarial Invariance}},
    author = {Jaiswal, Ayush and Wu, Yue and AbdAlmageed, Wael and Natarajan, Premkumar},
    journal = {arXiv preprint arXiv:1905.03629},
    year = {2019}
}
```


## Disclaimer

The code provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use.


## Comments

The code has been refactored for release purposes. Please contact us if something does not work or looks problematic.


## Contact

If you have any questions, drop an email to _ajaiswal@isi.edu_.

