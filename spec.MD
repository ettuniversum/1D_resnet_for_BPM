# Project Specification: 1D ResNet for PPG Heart Rate Estimation

## 1. Objective
Develop a 1D ResNet model in PyTorch to estimate heart rate (BPM) from Photoplethysmogram (PPG) signals collected from an Adafruit QtPy ESP32 sensor. The model will be exported to TFLite format for real-time inference on the Android "Wearout" application.

## 2. Technical Stack
- **Language:** Python 3.x
- **Deep Learning Framework:** PyTorch
- **Model Export:** TFLite (via ONNX conversion)
- **Data Source:** [Kaggle "5 Gamers" Dataset](https://www.kaggle.com/datasets/canaria/5-gamers)
- **Hardware:** Adafruit QtPy ESP32 (PPG Sensor)
- **Connectivity:** Bluetooth Low Energy (BLE)
- **Android Application:** [WearoutAndroid](https://github.com/ettuniversum/WearoutAndroid)

## 3. Scope of Work
- **Data Preprocessing:** Clean and preprocess PPG signal data from the Kaggle dataset to format suitable for 1D CNN training.
- **Model Development:** Design and train a 1D ResNet architecture to regress heart rate (BPM) from input PPG segments.
- **Model Optimization & Export:** Quantize and convert the trained PyTorch model to TFLite format suitable for mobile deployment.
- **Hardware Integration:** Ensure the model output aligns with the requirements of the `Wearout` Android application receiving data from the ESP32 IoT chip.

## 4. Pipeline
1.  **Data Ingestion & Cleaning:** Load dataset, handle noise/artifacts, normalize signals.
2.  **Model Training:** Define 1D ResNet architecture, train, and validate performance.
3.  **Model Conversion:** Convert PyTorch model to ONNX, then to TFLite.
4.  **Deployment:** Integrate the TFLite model into the `WearoutAndroid` project structure.

## 5. Directory Structure
```text
/
├── data/           # Raw and processed datasets
├── notebooks/      # Exploratory data analysis and training prototypes
├── src/            # Training scripts and utilities
├── models/         # Saved PyTorch and TFLite models
├── docs/           # Additional project documentation
└── spec.MD         # Project specification
```
