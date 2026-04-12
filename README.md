# IPL Win Predictor

This repository contains a polished Streamlit application that predicts Indian Premier League (IPL) match outcomes using a machine learning model. The app is built with Python and uses a Random Forest Classifier trained on historical IPL data to generate win probability insights.

Built with: **Python · Streamlit · Scikit-learn · Pandas · Matplotlib**

---

## Overview

The IPL Win Predictor lets users select two teams, choose a venue, and set toss details. The app then returns:

- a prediction for the winning team
- win probability scores for both sides
- a visual probability chart
- head-to-head historical stats
- model feature importance insights

This project is designed for data science learning, interactive reporting, and presenting a strong machine learning demo.

---

## What’s Included

- `app.py` — the main Streamlit web application
- `requirements.txt` — dependency list for Python
- `README.md` — project overview and usage instructions
- `SETUP_GUIDE.md` — step-by-step installation and deployment guide
- `matches.csv` — IPL dataset (must be downloaded separately)

---

## Key Features

- **Real-time prediction** of IPL match outcomes
- **Probability breakdown** for both teams
- **Head-to-head comparison** for selected matchups
- **Feature importance** visualization for model transparency
- **Modern UI theme** with custom styles and graphics

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Library | Scikit-learn |
| Input features | team1, team2, toss_winner, toss_decision, venue |
| Target | winner |
| Estimators | 200 |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ipl-predictor.git
cd ipl-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `matches.csv` from Kaggle and place it in the project root:

- https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set

### 4. Run the app

```bash
streamlit run app.py
```

Open the app at: `http://localhost:8501`

---

## Dataset

The app depends on the IPL match dataset from Kaggle. It includes historical match details, venues, toss results, and winning teams.

Primary data fields used:

- `team1`
- `team2`
- `venue`
- `toss_winner`
- `toss_decision`
- `winner`

---

## How It Works

1. Load the dataset from `matches.csv`
2. Clean and encode categorical data
3. Train a Random Forest model
4. Accept user input through the Streamlit interface
5. Return prediction results and visual analytics

---

## Recommended Workflow

1. Verify your Python version
2. Install packages from `requirements.txt`
3. Confirm `matches.csv` is present in the project root
4. Run `streamlit run app.py`
5. Explore predictions, charts, and analysis tabs

---

## Tech Stack

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## License

This project is released under the MIT License.
