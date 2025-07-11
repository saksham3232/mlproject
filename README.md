# End-to-End Machine Learning Project

This repository implements an end-to-end machine learning pipeline for predicting student exam performance based on demographic and test data. The project covers all steps including data preprocessing, model training, evaluation, and deployment via a web interface.

---

## Project Structure

```
mlproject/
├── application.py
├── requirements.txt
├── setup.py
├── src/
│   ├── utils.py
│   ├── logger.py
│   └── ... (other modules)
├── templates/
│   ├── home.html
│   └── index.html
├── notebook/
│   └── 2. MODEL TRAINING.ipynb
├── .ebextensions/
│   └── python.config
├── README.md
└── ... (other files/folders)
```

---

## Folders & Key Files

### [`application.py`](application.py)
Flask web application entry point. Defines routes for the home page and prediction. Handles form input and interacts with the ML prediction pipeline.

### [`requirements.txt`](requirements.txt)
Lists Python dependencies needed for the project, including ML and web libraries such as `scikit-learn`, `flask`, `xgboost`, `catboost`, etc.

### [`setup.py`](setup.py)
Standard Python package setup file for easy installation and dependency management.

### [`src/`](src/)
Main source code for the project:
- **utils.py**: Utility functions for saving/loading models and evaluating algorithms.
- **logger.py**: Logging setup for tracking pipeline events.
- *(Other modules likely exist for data ingestion, transformation, model training, prediction, exception handling, etc.)*

### [`templates/`](templates/)
HTML templates for the web interface:
- **home.html**: Form for user input to get predictions.
- **index.html**: Welcome page.

### [`notebook/`](notebook/)
Contains Jupyter Notebooks for EDA, model training, and experimentation.

### [`.ebextensions/python.config`](.ebextensions/python.config)
Configuration for deploying on AWS Elastic Beanstalk (Python container setup and WSGI path).

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/saksham3232/mlproject.git
cd mlproject
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python application.py
```
The web app will be available at `http://localhost:5000`.

---

## Project Workflow

1. **Data Collection & EDA:** Handled in Jupyter notebooks (see `notebook/`).
2. **Preprocessing & Feature Engineering:** Implemented in `src/` modules.
3. **Model Training:** Pipelines for model training, hyperparameter tuning, and evaluation.
4. **Model Serialization:** Save/load models using `dill` (see `utils.py`).
5. **Deployment:** Flask API for predictions; ready for cloud deployment (AWS EB config included).
6. **Web Interface:** Simple HTML forms for user input and displaying predictions.

---

## Example Usage

- Open the web app.
- Enter student demographic and exam info.
- Submit the form to get a predicted performance score.

---

## Contributing

Feel free to open issues or pull requests to enhance the project.

---

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [XGBoost](https://xgboost.ai/)
- [CatBoost](https://catboost.ai/)

---
