# Stock Day Trading Strategy

A comprehensive system for developing, testing, and implementing day trading strategies on Taiwan's stock market using machine learning techniques.

## Overview

This project implements a machine learning-based stock selection and day trading strategy system. It utilizes historical stock data from Taiwan's markets to train models that predict intraday price movements, specifically targeting stocks that could provide profitable day trading opportunities.

## Features

- **Flexible Feature Engineering**: Supports multiple feature types, including price-based, volume-based, margin trading, foreign investor data, broker activity, and market index features
- **Advanced ML Models**: Primary support for XGBoost with extensibility for other models (LightGBM, TabNet)
- **Efficient Caching System**: Implements a robust caching mechanism to save processed data and avoid redundant calculations
- **Market Index Integration**: Incorporates market index data to provide context for individual stock movements
- **Comprehensive Backtesting**: Tests strategy performance across historical data with detailed performance metrics
- **Optimized Data Processing**: Handles large datasets efficiently with memory usage optimization
- **Performance Visualization**: Generates equity curves and drawdown charts to evaluate strategy performance

## Program Flow

1. **Data Collection and Processing**:
   - Retrieves historical stock data with configurable lookback periods
   - Supports various data sources including stock prices, broker activities, margin trading, and foreign investor activities
   - Processes raw data into machine learning features

2. **Feature Engineering**:
   - Calculates technical indicators and derived features
   - Generates composite features combining different data sources
   - Creates market relative features based on index performance
   - Performs feature selection to identify most predictive variables

3. **Model Training**:
   - Prepares training data with proper scaling and preprocessing
   - Trains XGBoost classification model to predict profitable day trading opportunities
   - Evaluates model performance on validation data
   - Supports model persistence through caching for reuse

4. **Trading Strategy Backtesting**:
   - Applies trained model to test period data
   - Simulates day trading with configurable capital per trade
   - Implements entry/exit rules and risk management
   - Calculates profit/loss and performance metrics

5. **Performance Analysis**:
   - Generates detailed performance reports
   - Visualizes equity curves and drawdowns
   - Calculates financial metrics (Sharpe ratio, Sortino ratio)
   - Analyzes feature importance

## Installation and Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database with Taiwan stock market data
- Required Python packages (installed via Poetry)

### Installation with Poetry

1. Make sure you have Poetry installed. If not, install it following the instructions at [Python Poetry](https://python-poetry.org/docs/#installation).

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-day-trading.git
   cd stock-day-trading
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. Configure database connection in the code:
   ```python
   # Connection details are in connect_db() function
   conn = psycopg2.connect(
       host='your_host',
       database='your_database',
       user='your_username',
       password='your_password'
   )
   ```

## Usage

### Running with Poetry

1. **Basic run with default parameters**:
   ```bash
   poetry run python train_modify_v9.py
   ```

2. **Run with specific date ranges**:
   ```bash
   poetry run python train_modify_v9.py --train_start_date 2021-01-01 --train_end_date 2022-12-31 --test_start_date 2023-01-01 --test_end_date 2023-12-31
   ```

3. **Configure feature sets**:
   ```bash
   poetry run python train_modify_v9.py --use_price --use_volume --use_foreign --use_margin --use_composite
   ```

4. **Load existing model instead of training new one**:
   ```bash
   poetry run python train_modify_v9.py --load_model --model_path /path/to/your/model.json
   ```

5. **Adjust trading parameters**:
   ```bash
   poetry run python train_modify_v9.py --capital_per_trade 1000000 --threshold 0.65
   ```

### Command Line Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--train_start_date` | 2025-01-01 | Training period start date (YYYY-MM-DD) |
| `--train_end_date` | 2025-02-28 | Training period end date (YYYY-MM-DD) |
| `--test_start_date` | 2025-03-01 | Testing period start date (YYYY-MM-DD) |
| `--test_end_date` | 2025-03-14 | Testing period end date (YYYY-MM-DD) |
| `--use_price` | True | Include price-based features |
| `--use_volume` | True | Include volume-based features |
| `--use_broker` | False | Include broker activity features |
| `--use_foreign` | False | Include foreign investor features |
| `--use_margin` | False | Include margin trading features |
| `--use_composite` | True | Include composite features |
| `--model_type` | xgboost | Model type (xgboost, lightgbm, tabnet) |
| `--lookback_days` | 30 | Days to look back for calculating features |
| `--capital_per_trade` | 1000 | Capital allocated per trade (in thousands) |
| `--threshold` | 0.6 | Prediction probability threshold |
| `--load_model` | False | Load existing model instead of training |
| `--model_path` | None | Path to existing model file |

## Output

The program generates several output files in the `./analysis_results/` directory:

- Equity curve charts visualizing performance 
- Detailed trade logs
- Feature importance analysis
- Daily top stock selections
- Performance metrics reports

## Performance Metrics

The system evaluates strategy performance using the following metrics:

- Total profit
- Annualized return
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Average daily return

## Caching System

The program implements an efficient caching system to save:

- Raw data from database queries
- Processed features
- Trained models
- Prediction results

This significantly improves execution speed when running multiple tests with similar parameters.

## Directory Structure

- `analysis_results/`: Contains output files, charts, and analysis reports
- `models/`: Saved trained models
- `output_stock_data/`: External data files for stock lists
- `D:/data_cache/`: Cache storage location (configurable)

