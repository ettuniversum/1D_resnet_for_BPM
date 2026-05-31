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

## 4. Implementation Plan

### Phase 1: Data Preparation
- [ ] Implement robust data loader for Kaggle dataset.
- [ ] Apply band-pass filtering to remove motion artifacts and high-frequency noise.
- [ ] Segment PPG signals into windows (e.g., 5-10 seconds) with corresponding BPM labels.
- [ ] Split data into training, validation, and test sets.

### Phase 2: Model Design & Training
- [ ] Define 1D ResNet architecture tailored for time-series regression.
- [ ] Implement training loop with appropriate loss function (e.g., MSE or MAE).
- [ ] Experiment with hyperparameter tuning (learning rate, batch size, kernel sizes).
- [ ] Validate model performance against test set metrics (MAE, RMSE).

### Phase 3: Optimization & Deployment
- [ ] Convert PyTorch model to ONNX.
- [ ] Perform TFLite conversion with quantization (int8) to minimize footprint for mobile.
- [ ] Integrate TFLite model into the Android `Wearout` codebase.
- [ ] Conduct end-to-end testing with real-time BLE input.

## 5. Milestones
1. **Data Pipeline Ready:** Successfully load, clean, and visualize sample PPG data.
2. **Baseline Model:** Trained 1D ResNet achieving a defined performance metric (e.g., MAE < X BPM).
3. **Model Exported:** Successfully converted PyTorch model to a functioning TFLite model.
4. **Android Integration:** TFLite model running inference within the `WearoutAndroid` app.

## 6. Directory Structure
```text
/
├── data/           # Raw and processed datasets
├── notebooks/      # Exploratory data analysis and training prototypes
├── src/            # Training scripts and utilities
├── models/         # Saved PyTorch and TFLite models
├── docs/           # Additional project documentation
└── spec.MD         # Project specification
```
