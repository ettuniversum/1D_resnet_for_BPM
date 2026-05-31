# 1D ResNet for PPG Heart Rate Estimation

This repository contains the PyTorch implementation of a 1D ResNet model designed to estimate heart rate (Beats Per Minute - BPM) from Photoplethysmogram (PPG) signals. The model is prepared for deployment on mobile devices (Android) using TensorFlow Lite (TFLite) via ONNX conversion.

---

## Phase 1: Data Preprocessing Pipeline

We have implemented a robust time-series preprocessing pipeline in [data_processing.py](src/data_processing.py) to prepare raw data from the Kaggle "5 Gamers" dataset.

### Pipeline Details:
1. **Gap & Dropout Handling:** Detects time disconnections (sampling gaps $> 1.0$s or negative timeline jumps) in the raw logs. The session is automatically split into continuous chunks to prevent data windows from bridging disconnections.
2. **Band-pass Filtering:** Applies a 4th-order Butterworth band-pass filter (cutoffs: $0.5$ Hz to $4.0$ Hz, corresponding to $30$ to $240$ BPM) at $f_s = 100$ Hz to eliminate high-frequency noise and low-frequency motion artifacts.
3. **Sliding Window Segmentation:** Segments the continuous signal chunks into 10-second windows ($1000$ samples) with a 50% overlap ($5$-second step size).
4. **Ground-Truth BPM Estimation:**
   - Detects peaks (heartbeats) inside each window using Scipy's `find_peaks` with a minimum distance threshold ($25$ samples, corresponding to max HR of $240$ BPM) and prominence condition.
   - Calculates the average Inter-Beat Interval (IBI) between consecutive peaks.
   - Computes ground-truth BPM: $\text{BPM} = 60 \times f_s / \text{mean}(\text{IBI})$.
   - Validates that the estimated heart rate falls within a physiologically realistic range ($40$ to $200$ BPM). Windows with $< 5$ detected peaks are discarded.
5. **Z-Score Normalization:** Normalizes each window individually: $\text{window} \gets \frac{\text{window} - \mu}{\sigma}$ to make the signals invariant to sensor placement and baseline PPG amplitude variations.
6. **Subject-wise Dataset Splitting:** 
   To ensure the model generalizes to new individuals, the data is split subject-wise (by gamer) rather than randomly:
   - **Train Set (Gamers 1, 2, 3):** 36,399 segments
   - **Validation Set (Gamer 4):** 13,398 segments
   - **Test Set (Gamer 5):** 14,058 segments

---

## Dataset Structure

After running the preprocessing pipeline, the outputs are saved in `data/processed/` as NumPy arrays:
*   `train_x.npy` / `train_y.npy`: Preprocessed signals of shape `(36399, 1, 1000)` and their corresponding BPM labels of shape `(36399,)`.
*   `val_x.npy` / `val_y.npy`: Validation signals of shape `(13398, 1, 1000)` and BPM labels of shape `(13398,)`.
*   `test_x.npy` / `test_y.npy`: Test signals of shape `(14058, 1, 1000)` and BPM labels of shape `(14058,)`.

## How to Run Preprocessing

To configure the virtual environment and run the pipeline:
```bash
# Create and activate venv (using Python 3.9)
uv venv --python 3.9
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt scipy scikit-learn matplotlib

# Run the preprocessing pipeline
python src/data_processing.py
```
