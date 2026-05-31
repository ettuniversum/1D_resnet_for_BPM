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

---

## Phase 2: Model Design & Training

We designed a lightweight 10-layer 1D Residual Network (ResNet-1D) regression model suitable for mobile deployment. The model has `2,012,705` parameters and takes segmented, normalized PPG signals of shape `(batch_size, 1, 1000)` and outputs estimated BPM.

### How to Train the Model:
```bash
# Run training on GPU (or fallback to CPU)
python src/train.py --epochs 20 --batch_size 128 --lr 0.001 --device cuda
```
This saves the best-performing model checkpoint to `models/best_model.pth` and exports training curves to `models/loss_curves.png`.

---

## Phase 3: Model Export & Optimization

To prepare the trained model for real-time inference on the Android "Wearout" application, we convert the PyTorch model to ONNX format and then to TensorFlow Lite (TFLite) format with **Post-Training INT8 Quantization**.

### 1. Export PyTorch to ONNX
Export the PyTorch model (`models/best_model.pth`) to static ONNX format (`models/resnet10_5gamers.onnx`):
```bash
python src/export_onnx.py
```

### 2. Convert ONNX to TFLite (FP32 & INT8 Quantized)
First, convert the ONNX model to a TensorFlow `SavedModel` using `onnx2tf`. Then, compile it to FP32 and INT8-quantized TFLite formats:
```bash
python src/export_tflite.py
```
This script uses a representative dataset of 100 random windows from validation data (`val_x.npy`) to calibrate the activation ranges for INT8 quantization.

### Model Size Comparison:
*   **PyTorch Checkpoint:** `7.71 MB`
*   **ONNX Model:** `7.68 MB`
*   **TFLite FP32 Model:** `7.68 MB`
*   **TFLite INT8 Quantized Model:** `1.97 MB` (~74% footprint reduction)

---

## Android Deployment Walkthrough

Here is a step-by-step guide on how to integrate and run the quantized TFLite model inside the [WearoutAndroid](https://github.com/ettuniversum/WearoutAndroid) application.

### Step 1: Place the Model File
Copy the quantized model file `resnet10_5gamers_quant.tflite` from your models folder to your Android project's assets directory:
```text
app/src/main/assets/resnet10_5gamers_quant.tflite
```

### Step 2: Configure Dependencies
Add the TensorFlow Lite dependencies to your app's `build.gradle` file:
```groovy
dependencies {
    // TensorFlow Lite standard interpreter
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
}
```
Ensure that `.tflite` files are not compressed by the build system. Add the following block to your `build.gradle`:
```groovy
android {
    aaptOptions {
        noCompress "tflite"
    }
}
```

### Step 3: Implement the TFLite Classifier in Kotlin
Create a wrapper class to manage the model initialization, input/output buffer mapping, and inference execution:

```kotlin
import android.content.Context
import android.content.res.AssetFileDescriptor
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.io.File

class HeartRateEstimator(context: Context) {
    private var interpreter: Interpreter? = null
    
    companion object {
        private const val MODEL_NAME = "resnet10_5gamers_quant.tflite"
        private const val INPUT_LENGTH = 1000
    }

    init {
        val options = Interpreter.Options().apply {
            // Optional: Use GPU delegate or multiple threads if supported
            setNumThreads(2)
        }
        interpreter = Interpreter(loadModelFile(context, MODEL_NAME), options)
    }

    /**
     * Loads the TFLite model from the assets directory.
     */
    private fun loadModelFile(context: Context, modelName: String): java.nio.MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Estimates BPM from a 10-second preprocessed PPG signal window (1000 float samples).
     * 
     * @param ppgWindow Preprocessed (band-pass filtered & z-score normalized) signal array.
     * @return Estimated BPM (float).
     */
    fun estimateBPM(ppgWindow: FloatArray): Float {
        if (ppgWindow.size != INPUT_LENGTH) {
            throw IllegalArgumentException("Input signal must have exactly $INPUT_LENGTH samples (10 seconds at 100Hz).")
        }

        // TFLite expects shape: [batch_size, sequence_length, channels] -> [1, 1000, 1]
        // Create 3D input buffer: 1 batch, 1000 time steps, 1 channel
        val inputBuffer = Array(1) { Array(INPUT_LENGTH) { FloatArray(1) } }
        for (i in 0 until INPUT_LENGTH) {
            inputBuffer[0][i][0] = ppgWindow[i]
        }

        // TFLite output expects shape: [batch_size, output_features] -> [1, 1]
        val outputBuffer = Array(1) { FloatArray(1) }

        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)

        // Return regressed BPM
        return outputBuffer[0][0]
    }

    /**
     * Release TFLite resources when done.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
```

### Step 4: Run Inference in App Flow
Within your BLE data processing service, collect 10-second segments of preprocessed data (at 100Hz), pass them to the estimator, and publish the resulting BPM to the UI:

```kotlin
// Initialize estimator (usually in onCreate or Service setup)
val hrEstimator = HeartRateEstimator(applicationContext)

// In your data flow handler:
fun onNewWindowCollected(ppgData: FloatArray) {
    // 1. Ensure data is filtered and normalized exactly as done during training (Z-Score)
    // 2. Call the model
    val estimatedBpm = hrEstimator.estimateBPM(ppgData)
    
    // 3. Update UI/Log results
    Log.d("BPM_Inference", "Estimated Heart Rate: $estimatedBpm BPM")
}

// When clean up is needed (onDestroy)
hrEstimator.close()
```

