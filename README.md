# CKD Detection System

## Overview
The **Chronic Kidney Disease (CKD) Detection System** is a machine learning-based project that predicts CKD using medical data. This repository contains the necessary scripts for data preprocessing, model training, and inference.

## Installation and Setup
Follow these steps to install and run the CKD Detection System.

### Requirements
- **Operating System:** Windows (Linux is not supported)
- **Python:** Ensure you have Python installed

### 1. Clone the Repository
First, clone the repository to your local machine:

```bash
git clone https://github.com/Salahuddin-quadri/CKD.git
cd CKD
```

### 2. Set Up a Virtual Environment
It is recommended to create a virtual environment to manage dependencies:

```bash
python -m venv ckdenv
```

Activate the virtual environment:

```bash
ckdenv\Scripts\activate
```

### 3. Install Dependencies
Use `pip` to install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Run the Application
#### **(a) First, execute the main script to load the model**
```bash
python src/main.py
```

#### **(b) Then, execute the Flask application**
```bash
python frontend/app.py
```

### 5. Additional Notes
- Ensure that the necessary datasets are available in the `data/` directory.
- If the project requires pre-trained models, check the `models/` directory or train new models as needed.
- Modify configuration settings if required before running the application.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the Apache-2.0 License.


