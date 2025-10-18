# MLOps using Random Forest Model on the Iris Dataset

This project demonstrates MLOps practices using a Random Forest Classifier on the classic Iris dataset, with MLflow for experiment tracking and model management.

## Project Structure

```
MLOps/
├── myenv/                    # Python virtual environment
└── experiments/             
    └── randomforest.py      # Main model training script
```

## Features

- Random Forest Classifier implementation
- MLflow experiment tracking
- Comprehensive metrics logging (accuracy, precision, recall, F1-score)
- Feature importance analysis
- Automated model signature inference
- Error handling and logging
- Data validation

## Setup

1. Clone the repository:
```bash
git clone https://github.com/fahiiim/MLOps-using-the-Random-Forest-Model-on-the-Iris-Dataset.git
cd MLOps-using-the-Random-Forest-Model-on-the-Iris-Dataset
```

2. Create and activate virtual environment:
```bash
python -m venv myenv
# On Windows:
.\myenv\Scripts\activate
# On Unix/MacOS:
source myenv/bin/activate
```

3. Install dependencies:
```bash
pip install scikit-learn pandas mlflow numpy
```

## Usage

1. Run the model training script:
```bash
python experiments/randomforest.py
```

2. View MLflow UI (in a separate terminal):
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

## Model Performance

- Accuracy: 90%
- Precision: 90.24%
- Recall: 90%
- F1 Score: 89.97%

Detailed performance metrics for each class (setosa, versicolor, virginica) are logged in MLflow.

## MLflow Tracking

The project uses MLflow to track:
- Model parameters
- Performance metrics
- Model artifacts
- Feature importance
- Model signature

## License

This project is licensed under the MIT License - see the LICENSE file for details.