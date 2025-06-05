# P3Net: Systematic Deep Learning Model Selection for P300-Based Brain-Computer Interfaces

This repository contains the source code used in our research paper titled *"A Systematic Deep Learning Model Selection for P300-Based Brain-Computer Interfaces"*. The study explores the feasibility of conducting systematic model selection combined with mainstream deep learning architectures to construct accurate classifiers for decoding P300 event-related potentials.([github.com][1])

## Features

* **Diverse Model Architectures**: Implementation of various deep learning models, including CNNs, LSTMs, CNN-LSTM hybrids, EEGNet, and ShallowConvNet.
* **Comprehensive Evaluation**: Assessment of 232 CNNs (4 datasets × 58 structures), 36 LSTMs (4 datasets × 9 structures), and 320 CNN-LSTM models (4 datasets × 80 structures) of varying complexity.
* **Subject-Specific and Pooled Training**: Support for both subject-specific and pooled data training approaches.
* **Reproducibility**: Scripts and utilities to reproduce the results presented in the study.([github.com][1])

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/berdakh/P3Net.git
   cd P3Net
   ```



2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```



*Note: Ensure that [PyTorch](https://pytorch.org/) is installed, as it's central to the functionalities provided.*

## Usage

The repository includes various scripts for training and evaluating different model architectures:

* **Data Loading and Utilities**:

  * `nu_data_loader.py`: Functions for loading and preprocessing EEG data.
  * `nu_train_utils.py`: Utility functions for training models.([github.com][1])

* **Model Definitions**:

  * `nu_models.py`: Definitions of CNN, LSTM, CNN-LSTM, EEGNet, and ShallowConvNet architectures.([github.com][1])

* **Training Scripts**:

  * `train_CNN_pooled.py` / `train_CNN_subject_specific.py`: Training CNN models on pooled or subject-specific data.
  * `train_LSTM_pooled.py` / `train_LSTM_subject_specific.py`: Training LSTM models on pooled or subject-specific data.
  * `train_CNNLSTM_pooled.py` / `train_CNNLSTM_subject_specific.py`: Training CNN-LSTM hybrid models.
  * `train_EEGNET_subject_specific.py`: Training EEGNet models on subject-specific data.
  * `train_ShallowConvNet_subject_specific.py`: Training ShallowConvNet models on subject-specific data. 

*To execute a training script, navigate to the repository directory and run:*

```bash
python train_CNN_pooled.py
```



*Replace `train_CNN_pooled.py` with the desired training script.*

## Datasets

The study utilizes four publicly available EEG datasets for evaluating model performance. Please refer to the respective dataset sources for access and usage guidelines.

## Contribution

Contributions are welcome! If you have suggestions, bug reports, or enhancements, please open an issue or submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

This repository was developed by [Berdakh Abibullaev](https://github.com/berdakh), focusing on systematic deep learning model selection for P300-based brain-computer interfaces. 

---

*For detailed explanations and methodologies, refer to the research paper associated with this repository.*

If you need further assistance or have specific questions about any script or functionality, feel free to ask!
 
