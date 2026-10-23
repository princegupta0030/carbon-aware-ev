# Carbon-Aware Electric Vehicle Charging using Big Data Analytics and Renewable Energy Forecasting

## Project Overview
This project proposes a smart charging scheduling system for Electric Vehicles (EVs). It aims to reduce the carbon footprint and operational costs of charging by prioritizing periods when local solar energy generation is high. The project utilizes machine learning for renewable energy forecasting, big data tools for preprocessing, and mathematical optimization for generating charging schedules.

## Project Objective
The primary objective is to minimize grid carbon emissions associated with EV charging. This is achieved by forecasting solar energy availability and mathematically optimizing the charging schedule to align with these forecasts, while strictly adhering to user constraints such as battery capacity and departure deadlines.

## Key Features
*   **Automated Data Ingestion:** Programmatic download of Kaggle dataset via API.
*   **Big Data Processing:** Uses PySpark for data aggregation with a built-in Pandas fallback for systems with limited RAM.
*   **Renewable Energy Forecasting:** Implements a Random Forest regression model to predict solar generation based on weather data.
*   **Charging Optimization:** Uses linear programming (PuLP) to find the optimal charging schedule that meets user constraints.
*   **Interactive Dashboard:** A Streamlit-based web application to configure EV parameters and visualize charging strategies.

## How the System Works
1.  **Data Extraction:** Downloads historical weather and solar generation data.
2.  **Processing:** Cleans and aggregates 15-minute interval data into hourly intervals.
3.  **Forecasting:** Uses historical weather features (Irradiation and Temperature) to train a Random Forest model that predicts the AC Power output.
4.  **Optimization:** Taking the forecasted solar output, the system evaluates the effective carbon intensity of the grid for each hour. It then uses linear programming to schedule the exact kW of charging power required to reach the target charge before the departure deadline.
5.  **Visualization:** Compares the Carbon-Aware schedule against Immediate and Cheapest-Time schedules.

## Project Workflow
1. Download & Process Data -> 2. Train Forecasting Model -> 3. Define EV Constraints -> 4. Run Optimization -> 5. Compare Results

## Dataset
*   **Name:** `anikannal/solar-power-generation-data` (Solar Power Generation Data)
*   **Contents:** The dataset contains power generation data (DC and AC power) and weather sensor data (Ambient Temperature, Module Temperature, and Irradiation) from two solar power plants in India, collected every 15 minutes over a 34-day period.
*   **Download Method:** The dataset is downloaded programmatically using the `kaggle` Python API in `src/data_loader.py`.
*   **Limitations:** The dataset covers a relatively short time span (34 days) and lacks seasonal variation. Furthermore, the dataset does not include real-time dynamic grid emission factors or electricity pricing, requiring assumptions to be made for cost and carbon optimization.

## Technologies Used
*   **Programming Language:** Python 3
*   **Data Processing:** Pandas, PySpark
*   **Machine Learning:** scikit-learn (Random Forest)
*   **Optimization:** PuLP
*   **Web Dashboard:** Streamlit, Plotly

## Machine Learning / Forecasting Approach
The project uses a **Random Forest Regressor** to forecast solar AC power.
*   **Features:** Irradiation, Module Temperature, Ambient Temperature, and Hour of the Day.
*   **Data Split:** A strict chronological train/test split (80% Train, 20% Test) is used to prevent data leakage and simulate real-world forecasting.
*   **Metrics:** The model is evaluated on the unseen test set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Big Data Processing
The data preprocessing is handled in `src/data_processing.py`.
*   **Implementation:** The script uses PySpark to demonstrate big data transformations. It loads the generation and weather CSVs, standardizes timestamps, joins the datasets, and aggregates the 15-minute readings into hourly summaries. The PySpark session is constrained to use a 2GB driver memory limit to ensure stability on 8 GB RAM machines.
*   **Fallback:** If the PySpark session fails to initialize (e.g., due to Java dependency issues or memory limits), the code automatically catches the exception and falls back to a pure Pandas implementation that achieves the exact same aggregation.

## EV Charging Optimization
The project uses the `PuLP` library to schedule EV charging based on a defined timeframe and energy requirement. The system considers:
*   **Constraints:** Battery capacity, current charge, target charge, maximum charger power, and departure deadline (hours available).
*   **Strategies Compared:**
    1.  **Immediate Charging:** Charges the EV as quickly as possible at maximum power until the target is reached.
    2.  **Cheapest-Time Charging:** Minimizes cost based on an assumed Time-of-Use tariff (Off-peak: 5 Rs/kWh, Peak: 8 Rs/kWh).
    3.  **Carbon-Aware Charging:** Minimizes total emissions. The effective grid emission factor (base 0.82 kg CO2/kWh) is assumed to decrease linearly during hours where local forecasted solar generation is high.

## Dashboard
The `app.py` script runs an interactive Streamlit dashboard.
*   **Controls:** The sidebar contains buttons to trigger the data download, preprocessing, and model training. The main panel provides numerical inputs to configure the EV (Battery Capacity, Current Charge, Target Charge, Charger Power, and Hours Available).
*   **Results:** The dashboard displays the forecasting metrics (MAE and RMSE), a table comparing total cost and emissions across the three strategies, and an interactive Plotly bar chart visualizing the charging power allocated per hour for each strategy overlaid with the normalized solar forecast.

## Project Structure
```
carbon-aware-ev/
│
├── data/                      # Directory for raw and processed CSV data (git-ignored)
├── notebooks/                 # Directory for Jupyter notebooks
├── src/                       # Source code modules
│   ├── models/                # Saved ML models and prediction outputs (git-ignored)
│   ├── data_loader.py         # Kaggle API download script
│   ├── data_processing.py     # PySpark/Pandas preprocessing script
│   ├── forecasting.py         # Random Forest training and evaluation
│   └── optimization.py        # PuLP linear programming optimization
│
├── app.py                     # Streamlit dashboard application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd carbon-aware-ev
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Kaggle API Configuration
To automatically download the dataset, you must configure the Kaggle API.
1. Log in to Kaggle and navigate to your account settings.
2. Click "Create New API Token" to download a `kaggle.json` file.
3. Place this file in the appropriate directory for your OS:
   * **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   * **Linux/Mac:** `~/.kaggle/kaggle.json`
4. Ensure the file is kept secure and never committed to source control.

## How to Run
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser to the provided local URL (typically `http://localhost:8501`).
3. Use the sidebar buttons in order:
   * **1. Download & Process Data**
   * **2. Train Forecasting Model**
4. Configure your EV parameters in the main window.
5. Click **Run Optimization** to generate and compare the charging schedules.

## Forecasting Results
Based on the chronological test set evaluation (2020-06-11 to 2020-06-17):
*   **Mean Absolute Error (MAE):** 1020.96 kW
*   **Root Mean Squared Error (RMSE):** 1893.45 kW

## Results and Comparison
The optimization module outputs schedules that ensure the EV reaches its target charge before the user's departure. The Streamlit dashboard visually compares the results:
*   **Immediate Charging** usually incurs the highest cost and highest emissions, as it ignores solar availability and tariffs.
*   **Cheapest-Time Charging** successfully shifts demand to nighttime off-peak hours, minimizing Rs/kWh, but may rely entirely on coal-heavy grid power.
*   **Carbon-Aware Charging** shifts demand to midday hours aligning with the peak solar forecast, effectively minimizing the total kg of CO2 emitted.

## Limitations
*   The optimization currently relies on a static estimation formula linking gross solar plant output to local grid emission reductions.
*   Pricing tariffs are hard-coded assumptions for demonstration purposes, as real-time Indian dynamic pricing data is not available in the dataset.
*   The system is currently designed for a single EV user and does not model constraints for a fleet or commercial charging station.

## Future Scope
*   Integrate real-time dynamic pricing and grid emission API endpoints (e.g., Electricity Maps).
*   Extend the optimization model to handle multiple EVs sharing limited charging infrastructure.
*   Experiment with advanced forecasting models such as LSTMs or XGBoost.
