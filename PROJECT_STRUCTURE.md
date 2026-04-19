# PSL Score Predictor - End-to-End MLOps Project

A complete machine learning project structure for building, training, and deploying a PSL (Premier Soccer League) score prediction model.

## Project Structure

```
PSL-SCORE-PREDICTOR/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Clean data for modeling
│   └── external/         # Third-party data
├── src/
│   ├── data/            # Data loading and processing
│   ├── features/        # Feature engineering
│   ├── models/          # Model training and evaluation
│   └── utils/           # Helper functions
├── notebooks/           # Jupyter notebooks for exploration
├── models/              # Trained model artifacts
├── reports/             # Generated analysis and figures
├── tests/               # Unit and integration tests
├── config/              # Configuration files
├── requirements.txt     # Python dependencies
├── setup.py            # Package setup file
├── Makefile            # Common commands
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PSL-SCORE-PREDICTOR.git
cd PSL-SCORE-PREDICTOR
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
make install
```

For development with additional tools:
```bash
make install-dev
```

## Quick Start

### Data Preparation
Place raw data in `data/raw/` directory.

### Training
```bash
make train
```

### Testing
```bash
make test
```

### Code Quality
```bash
make lint
make format
```

## Key Modules

### `src/data/`
- **load_data.py**: Data loading and saving functions

### `src/features/`
- **feature_engineering.py**: Feature creation and preprocessing

### `src/models/`
- **train_model.py**: Model training and evaluation

### `src/utils/`
- **logger.py**: Logging configuration

## Using Docker

Build the image:
```bash
docker build -t psl-predictor .
```

Run with Docker Compose:
```bash
docker-compose up -d
```

Access Jupyter at `http://localhost:8888` and MLflow at `http://localhost:5000`.

## Configuration

Edit `config/config.py` to customize:
- Project directories
- Data settings (test size, random state)
- Model hyperparameters
- Logging level

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linters
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Your Name - [your.email@example.com](mailto:your.email@example.com)

## References

- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
