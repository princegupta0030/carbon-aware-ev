import os
import kaggle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_solar_data(download_path="data"):
    """
    Downloads the 'anikannal/solar-power-generation-data' dataset from Kaggle
    using the Kaggle API.

    Requires a valid kaggle.json file in the appropriate location
    (e.g., ~/.kaggle/kaggle.json on Linux/macOS or C:\\Users\\<User>\\.kaggle\\kaggle.json on Windows).
    """
    dataset_name = "anikannal/solar-power-generation-data"
    os.makedirs(download_path, exist_ok=True)

    # Check if data already exists to avoid redundant downloads
    expected_files = [
        "Plant_1_Generation_Data.csv",
        "Plant_1_Weather_Sensor_Data.csv",
        "Plant_2_Generation_Data.csv",
        "Plant_2_Weather_Sensor_Data.csv"
    ]

    all_files_exist = all(os.path.exists(os.path.join(download_path, f)) for f in expected_files)

    if all_files_exist:
        logger.info(f"Dataset already exists in {download_path}. Skipping download.")
        return True

    logger.info(f"Downloading dataset '{dataset_name}' to {download_path}...")
    try:
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        logger.info("Download and extraction complete.")
        return True
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.error("Please ensure you have configured your Kaggle API credentials (kaggle.json) correctly.")
        return False

if __name__ == "__main__":
    download_solar_data()
