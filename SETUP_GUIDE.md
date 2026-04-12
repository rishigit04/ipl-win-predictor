# IPL Win Predictor — Setup Guide

This guide walks you through installing and running the IPL Win Predictor app from scratch. It is written so any beginner can follow along and finish with a working Streamlit application.

---

## 1. Prerequisites

You will need:

- Python 3.10 or higher
- Internet access to download packages and the dataset
- A terminal or command prompt
- Optional: a GitHub account if you want to publish the project

---

## 2. Install Python

If Python is not already installed, download it from:

- https://python.org/downloads

During installation, make sure **Add Python to PATH** is enabled.

Verify the installation:

```bash
python --version
```

If you see a version number such as `Python 3.13.x`, you are ready.

---

## 3. Prepare the Project Folder

Open your terminal and run:

```bash
mkdir ipl-predictor
cd ipl-predictor
```

Then copy the project files into this folder:

- `app.py`
- `requirements.txt`
- `README.md`
- `SETUP_GUIDE.md`
- `.gitignore`

If you have a `logo.png`, add that file here too.

---

## 4. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

This will download everything needed for the app to run.

---

## 5. Download the Dataset

The app uses IPL match history data from Kaggle.

Download the dataset here:

- https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set

After downloading and extracting the zip file, place `matches.csv` in the `ipl-predictor` folder.

Your folder should look like:

```
ipl-predictor/
├── app.py
├── requirements.txt
├── README.md
├── SETUP_GUIDE.md
├── .gitignore
└── matches.csv
```

---

## 6. Run the App

From the project folder, start Streamlit:

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

If the browser does not open automatically, paste the URL into your browser.

---

## 7. Test the Application

Try the following:

- Choose two IPL teams
- Select a venue
- Pick the toss winner and decision
- Click **Predict Winner**

The app should display the predicted winner, probability charts, and head-to-head statistics.

---

## 8. Common Issues and Solutions

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: matches.csv` | Make sure `matches.csv` is in the same folder as `app.py` |
| `streamlit: command not found` | Run `python -m streamlit run app.py` |
| Port 8501 already in use | Use `streamlit run app.py --server.port 8502` |

---

## 9. Optional: Publish to GitHub

If you want to share the project, create a GitHub repository and push the folder using these commands:

```bash
git init
git add .
git commit -m "Add IPL Win Predictor project"
git remote add origin https://github.com/YOUR_USERNAME/ipl-predictor.git
git branch -M main
git push -u origin main
```

---

## 10. Optional: Deploy to Streamlit Cloud

If you want a live public URL, deploy the app on Streamlit Cloud:

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Create a new app
4. Set the repository and choose `app.py` as the main file

This will give you a public link you can share with others.

---

## 11. What You’ve Done

You now have a working IPL prediction app that:

- loads real IPL data
- trains a Random Forest model
- accepts user inputs in a clean UI
- displays win probabilities and match analytics

Great work! If you want, you can continue improving the app with team logos, additional statistics, or advanced model comparisons.
| Git push asks for password | Use a GitHub Personal Access Token instead of password |

---

## 💡 Tips to Impress Your Teacher

1. **Open with the live app** — don't start with slides, open the app immediately
2. **Let them interact** — ask your teacher to pick the teams
3. **Explain the model** — "It uses 200 decision trees and learned from 800+ real matches"
4. **Show the GitHub repo** — demonstrates real-world software practices
5. **Mention the feature importance chart** — shows you understand what the model learned

Good luck! 🏏
