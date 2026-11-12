# ☀️ Solar Sense – Chengalpattu
**Machine Learning for Daily Solar Irradiance Prediction & Climate Change Mitigation**  
*Course Project: 21CEO310T – Global Warming and Climate Change (Unit V: Mitigation & Unit II: Indian Climate System)*  
*Location: Chengalpattu, Tamil Nadu, India (12.69°N, 79.98°E)*

---

## 📌 Project Overview
Solar photovoltaic (PV) generation is an essential clean energy technology for mitigating anthropogenic global warming. In India's coal-heavy power grid, every kilowatt-hour (kWh) produced by rooftop or utility-scale solar directly offsets fossil fuel combustion, abating greenhouse gas (GHG) emissions. However, solar generation is intrinsically intermittent, driven by daily atmospheric variables like cloud optical depth, diurnal temperature swings, relative humidity, and monsoon rains.

**Solar Sense – Chengalpattu** is an interactive, production-grade Streamlit web application that lets students, researchers, and climate engineers:
1. **Verify ML against Real Data:** Query live NASA POWER satellite observations and compare ground reality with machine learning predictions.
2. **Explore Atmospheric Physics:** Simulate extreme heatwaves, Northeast Monsoon cloudbursts, and cyclones with interactive weather sliders and meteorological interpretations.
3. **Quantify Climate Impact:** Calculate clean AC electricity (kWh), avoided CO₂ emissions (tonnes), and equivalent mature trees for custom rooftop PV systems (default: 5 kWp).
4. **Inspect Model Explainability:** Understand feature importance through Random Forest permutation importance and pre-computed SHAP (SHapley Additive exPlanations) summary plots.

---

## 🏆 Benchmark Summary (Unseen 2023–2024 Test Set)

Models were trained on 8 years (2015–2022, 2,922 days) of NASA POWER daily data and evaluated strictly on 2 unseen test years (2023–2024, 731 days) using a time-based split to avoid temporal data leakage:

| Model | MAE (kWh/m²/day) | MSE | RMSE | R² Score | Key Hyperparameters / Details |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SVR (Support Vector Regression)** 🥇 | **0.560** | **0.583** | **0.764** | **0.673** | RBF kernel, `C=10`, `epsilon=0.2`, `gamma=0.05`, StandardScaler pipeline |
| **Random Forest Regressor** 🥈 | 0.562 | 0.561 | 0.749 | **0.686** | `n_estimators=400`, `max_depth=12`, `min_samples_leaf=5` |
| **Linear Regression** | 0.673 | 0.756 | 0.869 | 0.577 | Standardized OLS regression |
| **Baseline (Climatology)** | 0.788 | 1.047 | 1.023 | 0.414 | Historical mean irradiance per day-of-year (2015–2022) |

*Key Result:* The best model (**SVR**) reduces estimation error by **28.9%** over the climatological baseline, explaining **67.3%** of daily irradiance variance on unseen test years.

---

## 🛰️ Data Source & Features
* **Source:** [NASA POWER (Prediction Of Worldwide Energy Resources)](https://power.larc.nasa.gov/) daily single-point API (12.69°N, 79.98°E).
* **Target:** `ALLSKY_SFC_SW_DWN` — All Sky Surface Shortwave Downward Irradiance (kWh/m²/day).
* **Exact Engineered Features:**
  1. `T2M`: Daily average temperature at 2 meters (°C).
  2. `T_RANGE`: Diurnal temperature range (`T2M_MAX - T2M_MIN`, °C). Wide swings indicate clear, unclouded skies; narrow swings indicate cloud cover.
  3. `RH2M`: Relative humidity at 2 meters (%). High humidity correlates with cloudiness and optical attenuation.
  4. `WS2M`: Daily average wind speed at 2 meters (m/s).
  5. `PRECTOTCORR`: Daily total precipitation (mm/day). Primary indicator of dense cloud cover.
  6. `doy_sin`: $\sin(2\pi \cdot \text{dayofyear} / 365.25)$ (continuous seasonal solar declination cycle).
  7. `doy_cos`: $\cos(2\pi \cdot \text{dayofyear} / 365.25)$ (continuous seasonal solar declination cycle).

---

## 📱 Application Pages
1. **🏠 Home / Overview:** Project motivation, climate relevance (Unit V / Unit II), 4 headline metric cards, and 4-step pipeline strip.
2. **⚡ Live Check (Main Page):** Single-day and date-range (up to 31 days) verification against live NASA POWER API observations, absolute error badges, grouped Plotly charts, and CSV export. Includes automatic and manual offline fallback to `clean_data.csv`.
3. **🎛️ What-if Simulator:** Interactive weather sliders, preset meteorological scenarios (May Heatwave, Northeast Monsoon Storm, Winter Morning, Cyclone), multi-model comparison, and plain-English physical interpretation.
4. **📊 Model Performance:** Interactive metric tables (MAE, MSE, RMSE, R²), predicted-vs-actual scatter analysis with 1:1 parity line, error by month bar chart, and time-based split explanation.
5. **🔍 Explainability:** Random Forest permutation importance ranking, plain-language physical explanations, and pre-computed SHAP bee-swarm plot.
6. **🌱 Solar & CO₂ Calculator:** Custom system size (kWp), performance ratio (PR), and grid emission factor (EF) inputs; monthly generation profile; avoided coal combustion; and equivalent mature trees planted.
7. **ℹ️ About & Data:** NASA POWER data provenance, variables dictionary, quality assurance summary, academic limitations, formal citations, and artifact download buttons.

---

## 💻 Local Installation & Quickstart

### Prerequisites
* Python 3.10 to 3.13
* Git

### Step-by-Step Instructions
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/solar-sense-chengalpattu.git
cd solar-sense-chengalpattu

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install pinned dependencies (scikit-learn==1.6.1)
pip install -r requirements.txt

# 4. Run automated tests
pytest -v tests/

# 5. Launch Streamlit web app
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure
```
solar-sense-chengalpattu/
├── app.py                            # Streamlit main entrypoint & sidebar navigation
├── config.py                         # Central constants, paths, and metadata
├── requirements.txt                  # Pinned dependencies (scikit-learn==1.6.1)
├── solar_prediction_chengalpattu.ipynb # Full research & model training notebook
├── .gitignore                        # Git exclusion rules
├── .streamlit/
│   └── config.toml                   # Solar amber UI theme configuration
├── utils/
│   ├── __init__.py
│   ├── features.py                   # EXACT feature engineering: build_features()
│   ├── data.py                       # NASA POWER API client + 6h cache + offline fallback
│   └── models.py                     # Cached model loaders, inference & climate formulas
├── views/
│   ├── __init__.py
│   ├── p1_overview.py                # Home / Overview view
│   ├── p2_live.py                    # Live Check view
│   ├── p3_simulator.py               # What-if Simulator view
│   ├── p4_performance.py             # Model Performance view
│   ├── p5_explainability.py          # Explainability & SHAP view
│   ├── p6_calculator.py              # Solar & CO2 Calculator view
│   └── p7_about.py                   # About & Data view
├── artifacts/
│   ├── config.json                   # Hyperparameters, feature list, and metadata
│   ├── clean_data.csv                # 10-year cleaned daily dataset (2015-2024)
│   ├── metrics.csv                   # Benchmark evaluation metrics
│   ├── test_predictions.csv          # Unseen test set predictions (2023-2024)
│   ├── rf_permutation_importance.csv # Random Forest permutation importance scores
│   ├── shap_summary.png              # Pre-computed SHAP bee-swarm plot
│   ├── linear_regression.joblib      # Standardized linear regression pipeline (1 KB)
│   ├── svr.joblib                    # Tuned SVR pipeline with RBF kernel (76 KB)
│   └── random_forest.joblib          # Tuned Random Forest regressor (5.2 MB)
├── tests/
│   ├── __init__.py
│   └── test_app.py                   # Pytest test suite (6 passing unit tests)
└── screenshots/                      # Screenshots of all application pages
```

---

## 🔄 How to Retrain Models from Notebook
1. Open `solar_prediction_chengalpattu.ipynb` in Jupyter Notebook, VS Code, or Google Colab.
2. The notebook downloads raw data directly from NASA POWER for Chengalpattu, runs missing value interpolation and sanity checks, builds features, tunes hyperparameters with `TimeSeriesSplit(n_splits=5)`, evaluates test metrics, and exports all files into `./artifacts/`.
3. To update models in the app, simply replace the joblib and CSV files in `./artifacts/`.

---

## ⚠️ Academic Limitations (Course 21CEO310T)
1. **Spatial Resolution:** NASA POWER relies on a coarse ~0.5° × 0.5° (~50 km × 50 km) grid. Rooftop shading, local micro-climates, and localized sea-breezes are averaged out.
2. **Temporal Resolution:** Inputs are daily averages; high-frequency cloud ramps (sub-hourly) are not captured.
3. **Fixed Performance Ratio (PR):** Uses a static 0.75 derating factor, omitting dynamic soiling accumulation and cell temperature coefficients.
4. **National Emission Factor:** Uses the CEA Indian national grid average (0.71 tCO₂/MWh); actual state-level marginal emissions in Tamil Nadu vary with wind and solar penetration.
5. **Estimation vs. Forecasting:** Models perform same-day estimation given observed weather variables rather than forward day-ahead numerical weather forecasting.

---

## 🚀 Free Public Cloud Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended)
1. **Create GitHub Repo:**
   ```bash
   git init
   git add .
   git commit -m "feat: initial commit for Solar Sense Chengalpattu"
   git branch -M main
   git remote add origin https://github.com/<your-username>/solar-sense-chengalpattu.git
   git push -u origin main
   ```
2. **Sign In:** Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. **Deploy App:**
   * Click **New App**.
   * Repository: `<your-username>/solar-sense-chengalpattu`
   * Branch: `main`
   * Main file path: `app.py`
   * App URL: Customize your subdomain (e.g. `solar-sense-chengalpattu.streamlit.app`).
4. **Click Deploy!** The app will install requirements from `requirements.txt` and launch in 2–3 minutes.

### Option 2: Hugging Face Spaces (Alternative)
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Click **New Space** -> Choose **Streamlit** SDK.
3. Clone the Space repo locally, copy project files into it, commit, and push. Hugging Face builds and hosts the app on a free tier.

---

## 📚 References & Citations
1. **NASA POWER Project:** *Prediction Of Worldwide Energy Resources*, NASA Langley Research Center. [https://power.larc.nasa.gov](https://power.larc.nasa.gov)
2. **Central Electricity Authority (CEA):** *CO₂ Baseline Database for the Indian Power Sector*, Ministry of Power, Government of India, 2024.
3. **Pedregosa, F., et al.:** *Scikit-learn: Machine Learning in Python*, JMLR 12:2825–2830, 2011.
4. **Lundberg, S. M., & Lee, S.-I.:** *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017.

---
*Live App URL Placeholder:* `https://solar-sense-chengalpattu.streamlit.app`
