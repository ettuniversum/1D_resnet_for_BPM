import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=100, order=4):
    """Applies a bandpass filter to the PPG signal."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def process_gamer(gamer_prefix, raw_dir, fs=100):
    """Loads, cleans, and segments PPG files for a single gamer."""
    print(f"Processing data for {gamer_prefix}...")
    x_list = []
    y_list = []

    # List files matching gamer_prefix
    files = sorted(
        [
            f
            for f in os.listdir(raw_dir)
            if f.startswith(gamer_prefix) and f.endswith(".csv") and "ppg" in f
        ]
    )

    for filename in files:
        file_path = os.path.join(raw_dir, filename)
        print(f"  Reading {filename}...")
        df = pd.read_csv(file_path)

        # Handle empty/NaN values
        signal = df["Red_Signal"].ffill().bfill().values

        # Detect gaps > 1.0 second or time rollbacks
        times = pd.to_datetime("2000-01-01 " + df["Time"])
        time_diffs = np.diff(times.values) / np.timedelta64(1, "s")
        gap_indices = np.where((time_diffs > 1.0) | (time_diffs < 0.0))[0]

        # Split into continuous chunks
        start_idx = 0
        chunks = []
        for gap_idx in gap_indices:
            chunks.append(signal[start_idx : gap_idx + 1])
            start_idx = gap_idx + 1
        chunks.append(signal[start_idx:])

        print(f"  Split into {len(chunks)} continuous chunks.")

        # Process each chunk
        for chunk in chunks:
            if len(chunk) < 1000:
                continue

            # Apply bandpass filter
            filtered_chunk = bandpass_filter(chunk, fs=fs)

            # Segment into 10-second windows with 50% overlap
            for start in range(0, len(filtered_chunk) - 1000 + 1, 500):
                window = filtered_chunk[start : start + 1000]

                # Detect peaks
                peaks, _ = find_peaks(
                    window, distance=25, prominence=np.std(window) * 0.2
                )

                # If valid heart rate (between 5 and 40 peaks in 10s -> 30 to 240 BPM)
                if len(peaks) >= 5:
                    diffs = np.diff(peaks)
                    mean_ibi = np.mean(diffs) / fs
                    bpm = 60.0 / mean_ibi

                    if 40.0 <= bpm <= 200.0:
                        std_val = np.std(window)
                        if std_val > 1e-8:
                            window_norm = (window - np.mean(window)) / std_val
                            x_list.append(window_norm.reshape(1, -1))
                            y_list.append(bpm)

    if len(x_list) == 0:
        return (
            np.empty((0, 1, 1000), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Finished {gamer_prefix}. Extracted {len(y)} segments.")
    return x, y


def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Subject-wise splits
    train_gamers = ["gamer1", "gamer2", "gamer3"]
    val_gamers = ["gamer4"]
    test_gamers = ["gamer5"]

    print("=== Processing Train Split ===")
    train_x_parts, train_y_parts = [], []
    for gamer in train_gamers:
        x, y = process_gamer(gamer, raw_dir)
        if len(x) > 0:
            train_x_parts.append(x)
            train_y_parts.append(y)

    train_x = np.concatenate(train_x_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)
    print(f"Train shapes: X={train_x.shape}, Y={train_y.shape}")

    print("=== Processing Validation Split ===")
    val_x, val_y = process_gamer(val_gamers[0], raw_dir)
    print(f"Val shapes: X={val_x.shape}, Y={val_y.shape}")

    print("=== Processing Test Split ===")
    test_x, test_y = process_gamer(test_gamers[0], raw_dir)
    print(f"Test shapes: X={test_x.shape}, Y={test_y.shape}")

    # Save to disk
    np.save(os.path.join(processed_dir, "train_x.npy"), train_x)
    np.save(os.path.join(processed_dir, "train_y.npy"), train_y)
    np.save(os.path.join(processed_dir, "val_x.npy"), val_x)
    np.save(os.path.join(processed_dir, "val_y.npy"), val_y)
    np.save(os.path.join(processed_dir, "test_x.npy"), test_x)
    np.save(os.path.join(processed_dir, "test_y.npy"), test_y)

    print("=== Processing and Splitting Complete! ===")
    print(f"Files saved in {processed_dir}")


if __name__ == "__main__":
    main()

