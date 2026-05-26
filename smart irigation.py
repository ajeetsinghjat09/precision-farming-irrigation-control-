"""
Author  : Ajeet Singh Jaat
Project : Smart Irrigation & Rain Forecasting System
Hardware: DHT22 (temp + humidity) | Capacitive Soil Moisture Sensor
Cloud   : ThingSpeak IoT Dashboard

How it works:
  1. Historical weather CSV is loaded and transformed into rich features
  2. A Random Forest Classifier is trained — no if/else logic, pure ML
  3. Live sensor data is read every cycle and fed into the trained model
  4. Rain probability + label are pushed to ThingSpeak in real time
"""

import os, time, pickle, logging, warnings
import numpy as np
import pandas as pd
import requests

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.impute            import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
ENGINE_LOG = logging.getLogger("AgroRain")


# ═══════════════════════════════════════════════════
#   USER CONFIGURATION  ← edit only this section
# ═══════════════════════════════════════════════════

DATASET_PATH         = r"C:\Users\hp\Desktop\filtered_daily_weather_data.csv"
SAVED_MODEL_FILE     = "agro_model.pkl"
SAVED_STATE_FILE     = "agro_lagstate.pkl"

TS_WRITE_KEY         = "PXPUMQ4F30N0R48H"
TS_CHANNEL           = "2932567"
CYCLE_DELAY_SEC      = 15          # ThingSpeak free: min 15 s between writes

RAIN_CUTOFF_MM       = 1.0         # anything >= this is labelled "Rain"
RAIN_PROB_THRESHOLD  = 0.50        # model confidence needed to call "Rain"

SIMULATE_HW          = True        # False = read actual GPIO sensors


# ═══════════════════════════════════════════════════
#   PART A — DATA LOADING & FEATURE CONSTRUCTION
# ═══════════════════════════════════════════════════

def build_training_table(filepath: str) -> pd.DataFrame:
    """
    Reads the raw CSV, standardises column names, adds time-based
    features and rolling lag windows so the model can learn from
    recent weather trends — not just today's readings.
    """
    ENGINE_LOG.info("Reading dataset → %s", filepath)
    raw = pd.read_csv(filepath)
    raw.columns = raw.columns.str.strip()

    # Map whatever column names exist in the CSV to internal names
    col_map = {}
    for orig in raw.columns:
        low = orig.lower()
        if   "date"    in low:                       col_map[orig] = "rec_date"
        elif "rain"    in low:                       col_map[orig] = "rain_mm"
        elif "max"     in low and "temp" in low:     col_map[orig] = "t_high"
        elif "min"     in low and "temp" in low:     col_map[orig] = "t_low"
        elif "humid"   in low:                       col_map[orig] = "humidity"
        elif "soil"    in low or "moisture" in low:  col_map[orig] = "soil_wet"
    raw.rename(columns=col_map, inplace=True)

    needed = ["rec_date", "rain_mm", "t_high", "t_low"]
    absent = [c for c in needed if c not in raw.columns]
    if absent:
        raise ValueError(f"CSV missing these columns: {absent}  |  Found: {list(raw.columns)}")

    # Synthesise humidity + soil if sensor columns are absent in CSV
    if "humidity" not in raw.columns:
        ENGINE_LOG.warning("No humidity column found — synthesising for training.")
        np.random.seed(11)
        raw["humidity"] = np.clip(
            55 + 22 * np.sin(2 * np.pi * np.arange(len(raw)) / 365)
            + np.random.normal(0, 7, len(raw)), 20, 100
        )
    if "soil_wet" not in raw.columns:
        ENGINE_LOG.warning("No soil moisture column found — synthesising for training.")
        np.random.seed(33)
        raw["soil_wet"] = np.clip(
            raw["rain_mm"].shift(1).fillna(0) * 1.8
            + np.random.normal(38, 12, len(raw)), 0, 100
        )

    # Parse, sort, clean
    raw["rec_date"] = pd.to_datetime(raw["rec_date"], dayfirst=True, errors="coerce")
    raw.dropna(subset=["rec_date"], inplace=True)
    raw.sort_values("rec_date", inplace=True)
    raw.reset_index(drop=True, inplace=True)
    raw.drop_duplicates(inplace=True)

    # Fill remaining numeric nulls with column mean
    num_cols = ["rain_mm", "t_high", "t_low", "humidity", "soil_wet"]
    raw[num_cols] = SimpleImputer(strategy="mean").fit_transform(raw[num_cols])

    # Binary target: did it rain today?
    raw["rained"] = (raw["rain_mm"] >= RAIN_CUTOFF_MM).astype(int)

    # Calendar decomposition
    raw["cal_month"]  = raw["rec_date"].dt.month
    raw["cal_doy"]    = raw["rec_date"].dt.dayofyear
    raw["cal_season"] = (raw["cal_month"] % 12 // 3) + 1   # 1=Winter 2=Spring 3=Summer 4=Autumn

    # Derived sensor features
    raw["temp_swing"]    = raw["t_high"] - raw["t_low"]
    raw["heat_hum_idx"]  = raw["t_high"] * raw["humidity"] / 100.0

    # Lag features — yesterday, 2 days ago, 3 days ago
    for step in [1, 2, 3]:
        raw[f"rain_prev{step}"]  = raw["rain_mm"].shift(step)
        raw[f"hum_prev{step}"]   = raw["humidity"].shift(step)
        raw[f"soil_prev{step}"]  = raw["soil_wet"].shift(step)

    # 3-day rolling summaries
    raw["rain_3d_total"]  = raw["rain_mm"].shift(1).rolling(3).sum()
    raw["hum_3d_mean"]    = raw["humidity"].shift(1).rolling(3).mean()
    raw["soil_3d_mean"]   = raw["soil_wet"].shift(1).rolling(3).mean()

    raw.dropna(inplace=True)
    rain_pct = raw["rained"].mean() * 100
    ENGINE_LOG.info("Training table built → %d rows | Rain days: %.1f%%", len(raw), rain_pct)
    return raw


# ═══════════════════════════════════════════════════
#   PART B — MODEL TRAINING
# ═══════════════════════════════════════════════════

INPUT_FEATURES = [
    "t_high", "t_low", "temp_swing", "heat_hum_idx",
    "humidity", "soil_wet",
    "cal_month", "cal_doy", "cal_season",
    "rain_prev1", "rain_prev2", "rain_prev3",
    "hum_prev1",  "hum_prev2",  "hum_prev3",
    "soil_prev1", "soil_prev2", "soil_prev3",
    "rain_3d_total", "hum_3d_mean", "soil_3d_mean",
]


def train_and_save(weather_table: pd.DataFrame):
    """
    Scales features and trains a Random Forest Classifier.
    Prints accuracy, ROC-AUC, confusion matrix, and top feature importances.
    Saves model + scaler to disk so training only needs to run once.
    """
    sensor_matrix = weather_table[INPUT_FEATURES].values
    rain_labels   = weather_table["rained"].values

    feat_scaler   = StandardScaler()
    scaled_matrix = feat_scaler.fit_transform(sensor_matrix)

    tr_X, te_X, tr_y, te_y = train_test_split(
        scaled_matrix, rain_labels, test_size=0.2, stratify=rain_labels, random_state=17
    )

    # Random Forest — class_weight='balanced' handles rain/no-rain imbalance
    forest = RandomForestClassifier(
        n_estimators     = 250,
        max_depth        = 14,
        min_samples_split= 4,
        min_samples_leaf = 2,
        class_weight     = "balanced",
        random_state     = 17,
        n_jobs           = -1
    )

    ENGINE_LOG.info("Training Random Forest on %d samples …", len(tr_X))
    forest.fit(tr_X, tr_y)

    # ── Evaluation ──────────────────────────────────────────────────
    pred_labels = forest.predict(te_X)
    pred_probs  = forest.predict_proba(te_X)[:, 1]

    ENGINE_LOG.info("\n%s", "=" * 55)
    ENGINE_LOG.info("Test Accuracy : %.4f", accuracy_score(te_y, pred_labels))
    ENGINE_LOG.info("ROC-AUC Score : %.4f", roc_auc_score(te_y, pred_probs))
    ENGINE_LOG.info("\n%s", classification_report(te_y, pred_labels,
                             target_names=["No Rain", "Rain"]))
    ENGINE_LOG.info("Confusion Matrix:\n%s", confusion_matrix(te_y, pred_labels))

    cv_folds  = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    cv_scores = cross_val_score(forest, scaled_matrix, rain_labels,
                                 cv=cv_folds, scoring="roc_auc", n_jobs=-1)
    ENGINE_LOG.info("5-Fold CV AUC : %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    ENGINE_LOG.info("=" * 55)

    # Feature importance ranking
    importance_series = pd.Series(forest.feature_importances_, index=INPUT_FEATURES)
    ENGINE_LOG.info("\nTop 10 Predictors:\n%s",
                    importance_series.sort_values(ascending=False).head(10).to_string())

    # ── Save model + scaler ─────────────────────────────────────────
    with open(SAVED_MODEL_FILE, "wb") as mf:
        pickle.dump({"forest": forest, "scaler": feat_scaler}, mf)
    ENGINE_LOG.info("Model saved → %s", SAVED_MODEL_FILE)

    # Save the last known lag window from the training data
    last = weather_table.iloc[-1]
    lag_window = {
        "rain_prev1": weather_table["rain_mm"].iloc[-1],
        "rain_prev2": weather_table["rain_mm"].iloc[-2],
        "rain_prev3": weather_table["rain_mm"].iloc[-3],
        "hum_prev1":  weather_table["humidity"].iloc[-1],
        "hum_prev2":  weather_table["humidity"].iloc[-2],
        "hum_prev3":  weather_table["humidity"].iloc[-3],
        "soil_prev1": weather_table["soil_wet"].iloc[-1],
        "soil_prev2": weather_table["soil_wet"].iloc[-2],
        "soil_prev3": weather_table["soil_wet"].iloc[-3],
        "rain_3d_total": weather_table["rain_3d_total"].iloc[-1],
        "hum_3d_mean":   weather_table["hum_3d_mean"].iloc[-1],
        "soil_3d_mean":  weather_table["soil_3d_mean"].iloc[-1],
        "cal_month":  last["cal_month"],
        "cal_doy":    last["cal_doy"],
        "cal_season": last["cal_season"],
    }
    with open(SAVED_STATE_FILE, "wb") as sf:
        pickle.dump(lag_window, sf)
    ENGINE_LOG.info("Lag state saved → %s", SAVED_STATE_FILE)

    return forest, feat_scaler, lag_window


def load_saved_model():
    """Loads previously trained model + scaler + lag window from disk."""
    if not os.path.exists(SAVED_MODEL_FILE):
        raise FileNotFoundError(
            f"No saved model at '{SAVED_MODEL_FILE}'. Run training first."
        )
    with open(SAVED_MODEL_FILE, "rb") as mf:
        bundle = pickle.load(mf)
    with open(SAVED_STATE_FILE, "rb") as sf:
        lag_window = pickle.load(sf)
    ENGINE_LOG.info("Loaded model from disk → %s", SAVED_MODEL_FILE)
    return bundle["forest"], bundle["scaler"], lag_window


# ═══════════════════════════════════════════════════
#   PART C — SENSOR READING (DHT22 + SOIL)
# ═══════════════════════════════════════════════════

def dht22_read():
    """
    Reads temperature (°C) and humidity (%) from DHT22.
    Wiring: DHT22 DATA pin → GPIO 4 (BCM numbering)
    Install: pip install adafruit-circuitpython-dht
    """
    try:
        import board, adafruit_dht
        sensor = adafruit_dht.DHT22(board.D4)
        t = sensor.temperature
        h = sensor.humidity
        sensor.exit()
        if t is None or h is None:
            raise ValueError("DHT22 returned None values")
        ENGINE_LOG.info("DHT22 reading → Temp %.1f °C | Humidity %.1f %%", t, h)
        return float(t), float(h)
    except Exception as err:
        ENGINE_LOG.error("DHT22 error: %s", err)
        return None, None


def soil_sensor_read():
    """
    Reads soil moisture from capacitive sensor via MCP3008 ADC.
    Wiring: Sensor AOUT → MCP3008 Channel 0
    Install: pip install adafruit-mcp3xxx

    Calibration — measure these two values with YOUR sensor:
      COMPLETELY_DRY  : ADC value when sensor is in open air
      FULLY_WET       : ADC value when sensor tip is in water
    """
    COMPLETELY_DRY = 750    # ← calibrate this for your sensor
    FULLY_WET      = 290    # ← calibrate this for your sensor

    try:
        import busio, digitalio, board
        import adafruit_mcp3xxx.mcp3008 as MCP
        from adafruit_mcp3xxx.analog_in import AnalogIn

        spi_bus  = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
        chip_sel = digitalio.DigitalInOut(board.D5)
        adc_chip = MCP.MCP3008(spi_bus, chip_sel)
        ch0      = AnalogIn(adc_chip, MCP.P0)
        raw_val  = ch0.value >> 6   # scale 16-bit → 10-bit

        pct = (COMPLETELY_DRY - raw_val) / (COMPLETELY_DRY - FULLY_WET) * 100
        pct = float(np.clip(pct, 0.0, 100.0))
        ENGINE_LOG.info("Soil sensor reading → Raw %d | Moisture %.1f %%", raw_val, pct)
        return pct
    except Exception as err:
        ENGINE_LOG.error("Soil sensor error: %s", err)
        return None


def hardware_readings():
    """Reads both sensors and returns (temp, humidity, soil_moisture)."""
    temp, hum  = dht22_read()
    soil       = soil_sensor_read()
    return temp, hum, soil


def simulated_readings():
    """
    Generates realistic synthetic sensor values for testing
    without physical hardware. Uses diurnal variation patterns.
    """
    hr   = time.localtime().tm_hour
    temp = round(27 + 6 * np.sin(np.pi * hr / 12) + np.random.uniform(-1.2, 1.2), 1)
    hum  = round(np.clip(63 + 18 * np.cos(np.pi * hr / 12) + np.random.uniform(-6, 6), 25, 99), 1)
    soil = round(np.clip(44 + np.random.uniform(-12, 12), 5, 95), 1)
    ENGINE_LOG.info("Simulated → Temp %.1f °C | Humidity %.1f %% | Soil %.1f %%", temp, hum, soil)
    return temp, hum, soil


# ═══════════════════════════════════════════════════
#   PART D — REAL-TIME INFERENCE
# ═══════════════════════════════════════════════════

def assemble_live_row(temp: float, hum: float, soil: float, lag: dict) -> np.ndarray:
    """
    Constructs the exact same 21-feature vector the model was trained on,
    using current sensor values and the rolling lag window.
    """
    now     = pd.Timestamp.now()
    month   = now.month
    doy     = now.day_of_year
    season  = (month % 12 // 3) + 1

    t_high = temp + 2.5
    t_low  = temp - 2.5

    live_row = {
        "t_high":       t_high,
        "t_low":        t_low,
        "temp_swing":   t_high - t_low,
        "heat_hum_idx": t_high * hum / 100.0,
        "humidity":     hum,
        "soil_wet":     soil,
        "cal_month":    month,
        "cal_doy":      doy,
        "cal_season":   season,
        "rain_prev1":   lag.get("rain_prev1", 0),
        "rain_prev2":   lag.get("rain_prev2", 0),
        "rain_prev3":   lag.get("rain_prev3", 0),
        "hum_prev1":    lag.get("hum_prev1", hum),
        "hum_prev2":    lag.get("hum_prev2", hum),
        "hum_prev3":    lag.get("hum_prev3", hum),
        "soil_prev1":   lag.get("soil_prev1", soil),
        "soil_prev2":   lag.get("soil_prev2", soil),
        "soil_prev3":   lag.get("soil_prev3", soil),
        "rain_3d_total":lag.get("rain_3d_total", 0),
        "hum_3d_mean":  lag.get("hum_3d_mean", hum),
        "soil_3d_mean": lag.get("soil_3d_mean", soil),
    }
    return np.array([[live_row[feat] for feat in INPUT_FEATURES]])


def run_inference(forest, scaler, temp, hum, soil, lag):
    """
    Passes live sensor data through the scaler and forest model.
    Returns rain probability (0.0–1.0) and binary label (0 or 1).
    """
    row_raw    = assemble_live_row(temp, hum, soil, lag)
    row_scaled = scaler.transform(row_raw)
    rain_prob  = float(forest.predict_proba(row_scaled)[0][1])
    rain_flag  = int(rain_prob >= RAIN_PROB_THRESHOLD)
    return round(rain_prob, 4), rain_flag


def advance_lag_window(current_lag: dict, hum: float, soil: float,
                        rain_today_mm: float = 0.0) -> dict:
    """
    Shifts the lag window forward by one cycle.
    Called after every sensor reading so the next inference
    uses the most recent 3 readings as context.
    """
    updated = current_lag.copy()

    updated["rain_prev3"]   = current_lag.get("rain_prev2", 0)
    updated["rain_prev2"]   = current_lag.get("rain_prev1", 0)
    updated["rain_prev1"]   = rain_today_mm
    updated["rain_3d_total"]= updated["rain_prev1"] + updated["rain_prev2"] + updated["rain_prev3"]

    updated["hum_prev3"]    = current_lag.get("hum_prev2", hum)
    updated["hum_prev2"]    = current_lag.get("hum_prev1", hum)
    updated["hum_prev1"]    = hum
    updated["hum_3d_mean"]  = np.mean([updated["hum_prev1"], updated["hum_prev2"], updated["hum_prev3"]])

    updated["soil_prev3"]   = current_lag.get("soil_prev2", soil)
    updated["soil_prev2"]   = current_lag.get("soil_prev1", soil)
    updated["soil_prev1"]   = soil
    updated["soil_3d_mean"] = np.mean([updated["soil_prev1"], updated["soil_prev2"], updated["soil_prev3"]])

    return updated


# ═══════════════════════════════════════════════════
#   PART E — THINGSPEAK UPLINK
# ═══════════════════════════════════════════════════

def push_to_thingspeak(temp: float, hum: float, soil: float,
                        rain_prob: float, rain_flag: int):
    """
    Writes all 5 fields to the ThingSpeak channel.
      field1 = Temperature     (°C)
      field2 = Humidity        (%)
      field3 = Soil Moisture   (%)
      field4 = Rain Probability (0.00 – 1.00)
      field5 = Rain Label      (0 = No Rain | 1 = Rain)
    """
    endpoint = (
        f"https://api.thingspeak.com/update"
        f"?api_key={TS_WRITE_KEY}"
        f"&field1={temp}"
        f"&field2={hum}"
        f"&field3={soil}"
        f"&field4={rain_prob}"
        f"&field5={rain_flag}"
    )
    try:
        response = requests.get(endpoint, timeout=10)
        if response.status_code == 200 and response.text.strip() != "0":
            ENGINE_LOG.info(
                "ThingSpeak ✓  Entry #%s | T=%.1f  H=%.1f  S=%.1f  P(rain)=%.2f  Flag=%d",
                response.text.strip(), temp, hum, soil, rain_prob, rain_flag
            )
        else:
            ENGINE_LOG.warning("ThingSpeak rejected the write. Status=%s Body=%s",
                               response.status_code, response.text.strip())
    except requests.RequestException as conn_err:
        ENGINE_LOG.error("ThingSpeak connection error: %s", conn_err)


# ═══════════════════════════════════════════════════
#   MAIN — TRAIN ONCE → LOOP FOREVER
# ═══════════════════════════════════════════════════

def main():
    # ── Phase 1: Training or loading ────────────────────────────────
    if os.path.exists(SAVED_MODEL_FILE) and os.path.exists(SAVED_STATE_FILE):
        ENGINE_LOG.info("Saved model found — skipping training.")
        forest, scaler, rolling_lag = load_saved_model()
    else:
        ENGINE_LOG.info("No saved model found — building from CSV data …")
        weather_df               = build_training_table(DATASET_PATH)
        forest, scaler, rolling_lag = train_and_save(weather_df)

    # ── Phase 2: Real-time prediction loop ──────────────────────────
    ENGINE_LOG.info("\n%s", "=" * 55)
    ENGINE_LOG.info("AgroRain engine started  |  Sim mode: %s", SIMULATE_HW)
    ENGINE_LOG.info("Press Ctrl + C to stop")
    ENGINE_LOG.info("%s\n", "=" * 55)

    while True:
        try:
            # Read sensors
            if SIMULATE_HW:
                live_temp, live_hum, live_soil = simulated_readings()
            else:
                live_temp, live_hum, live_soil = hardware_readings()

            # Skip cycle if any sensor failed
            if None in (live_temp, live_hum, live_soil):
                ENGINE_LOG.warning("Incomplete sensor data — retrying next cycle.")
                time.sleep(CYCLE_DELAY_SEC)
                continue

            # Predict
            rain_probability, rain_flag = run_inference(
                forest, scaler, live_temp, live_hum, live_soil, rolling_lag
            )
            outcome = "RAIN EXPECTED" if rain_flag == 1 else "NO RAIN"

            print(f"\n{'━' * 48}")
            print(f"  Temperature    : {live_temp:.1f} °C")
            print(f"  Humidity       : {live_hum:.1f} %")
            print(f"  Soil Moisture  : {live_soil:.1f} %")
            print(f"  Rain Probability: {rain_probability * 100:.1f} %")
            print(f"  Prediction     : {outcome}")
            print(f"{'━' * 48}\n")

            # Push to ThingSpeak
            push_to_thingspeak(live_temp, live_hum, live_soil,
                                rain_probability, rain_flag)

            # Advance the lag window using current readings
            rolling_lag = advance_lag_window(rolling_lag, live_hum, live_soil)

            time.sleep(CYCLE_DELAY_SEC)

        except KeyboardInterrupt:
            ENGINE_LOG.info("AgroRain engine stopped by user.")
            break
        except Exception as unexpected:
            ENGINE_LOG.error("Unhandled error: %s", unexpected)
            time.sleep(CYCLE_DELAY_SEC)


if __name__ == "__main__":
    main()
