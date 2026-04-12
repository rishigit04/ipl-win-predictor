import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px
import base64
import warnings
warnings.filterwarnings('ignore')

# Page configuration and app metadata
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to load a local image file and convert it to base64
# This lets the app safely embed images as inline CSS backgrounds.
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Logo file '{image_path}' not found. Continuing without background logo.")
        return ""

# Get your logo as base64
logo_base64 = get_base64_image("logo.png")

# Load fonts, icons, and custom CSS styling for the app
st.markdown(f"""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ============================================
   CORE STYLES & RESET
   ============================================ */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
}}

.stApp {{
    background: linear-gradient(145deg, #0a0a0f 0%, #12121a 50%, #0d0d14 100%);
    color: #e0e0e0;
    min-height: 100vh;
    position: relative;
    animation: fadeInApp 1.5s ease-out;
}}

/* Logo Background Watermark - YOUR LOCAL IMAGE */
.stApp::before {{
    content: '';
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 600px;
    height: 600px;
    background-image: url('data:image/png;base64,{logo_base64}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    opacity: 0.05;
    pointer-events: none;
    z-index: 0;
}}

.stApp > div {{
    position: relative;
    z-index: 1;
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700;
}}

/* ============================================
   ADVANCED ANIMATIONS
   ============================================ */

/* Floating Animation */
@keyframes floatUp {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-15px); }}
}}

/* Pulse Glow Animation */
@keyframes pulseGlow {{
    0%, 100% {{ 
        box-shadow: 0 0 20px rgba(249,168,37,0.3), 0 0 40px rgba(249,168,37,0.1);
    }}
    50% {{ 
        box-shadow: 0 0 40px rgba(249,168,37,0.6), 0 0 80px rgba(249,168,37,0.3);
    }}
}}

/* Gradient Text Animation */
@keyframes gradientText {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* Rotate Animation */
@keyframes rotate360 {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}

/* Slide In From Bottom */
@keyframes slideInUp {{
    from {{
        opacity: 0;
        transform: translateY(40px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

/* Slide In From Left */
@keyframes slideInLeft {{
    from {{
        opacity: 0;
        transform: translateX(-40px);
    }}
    to {{
        opacity: 1;
        transform: translateX(0);
    }}
}}

/* Slide In From Right */
@keyframes slideInRight {{
    from {{
        opacity: 0;
        transform: translateX(40px);
    }}
    to {{
        opacity: 1;
        transform: translateX(0);
    }}
}}

/* Bounce Animation */
@keyframes bounce2 {{
    0%, 100% {{ transform: translateY(0); }}
    25% {{ transform: translateY(-8px); }}
    50% {{ transform: translateY(-4px); }}
    75% {{ transform: translateY(-6px); }}
}}

/* Shimmer Effect */
@keyframes shimmer {{
    0% {{ left: -100%; }}
    100% {{ left: 200%; }}
}}

/* Blinking Lights */
@keyframes blinkLights {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
}}

/* Staggered Fade In */
@keyframes fadeInDelay {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* Scale In Animation */
@keyframes scaleIn {{
    from {{ transform: scale(0.8); opacity: 0; }}
    to {{ transform: scale(1); opacity: 1; }}
}}

/* Wave Animation */
@keyframes wave {{
    0% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-5px); }}
    100% {{ transform: translateY(0); }}
}}

/* Spin Animation */
@keyframes fadeInApp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* ============================================
   HERO SECTION
   ============================================ */
.hero-section {{
    text-align: center;
    padding: 2rem 1rem;
    position: relative;
    overflow: hidden;
}}

.hero-section::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(249,168,37,0.08) 0%, transparent 50%);
    animation: rotate360 30s linear infinite;
    z-index: 0;
}}

.hero-icon {{
    position: relative;
    z-index: 1;
    display: inline-block;
    animation: floatUp 3s ease-in-out infinite;
}}

.hero-icon i {{
    font-size: 4rem;
    background: linear-gradient(135deg, #f9a825, #ff6f00, #f9a825);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientText 3s ease infinite;
}}

.hero-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f9a825 0%, #ff6f00 50%, #ffd54f 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 4px;
    margin: 1rem 0 0.5rem;
    position: relative;
    z-index: 1;
    animation: gradientText 4s ease infinite;
}}

.hero-subtitle {{
    color: #888;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
    animation: fadeInDelay 1s ease-out 0.3s both;
}}

/* ============================================
   STATS CARDS
   ============================================ */
.stats-container {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin: 2rem 0;
    animation: slideInUp 0.8s ease-out 0.2s both;
}}

.stat-card {{
    background: linear-gradient(145deg, #1a1a24, #141418);
    border: 1px solid #2a2a3a;
    border-radius: 16px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}

.stat-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 300%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(249,168,37,0.1),
        transparent
    );
    transition: left 0.5s;
}}

.stat-card:hover {{
    border-color: #f9a825;
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 15px 40px rgba(249,168,37,0.25);
}}

.stat-card:hover::before {{
    animation: shimmer 1.5s;
}}

.stat-icon {{
    font-size: 1.8rem;
    color: #f9a825;
    margin-bottom: 0.8rem;
    animation: bounce2 2s ease-in-out infinite;
}}

.stat-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg, #f9a825, #ffd54f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.stat-label {{
    color: #666;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}}

/* ============================================
   FEATURE PILLS
   ============================================ */
.feature-pills {{
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 2rem 0;
    animation: slideInUp 0.8s ease-out 0.4s both;
}}

.feature-pill {{
    background: linear-gradient(145deg, #15151c, #1a1a24);
    border: 1px solid #2a2a3a;
    border-radius: 30px;
    padding: 0.8rem 1.5rem;
    color: #888;
    font-size: 0.9rem;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: center;
    gap: 0.6rem;
    cursor: default;
}}

.feature-pill i {{
    color: #f9a825;
    font-size: 1rem;
}}

.feature-pill:hover {{
    background: linear-gradient(145deg, #1f1f2a, #252530);
    border-color: #f9a825;
    color: #f9a825;
    transform: translateY(-4px) scale(1.05);
    box-shadow: 0 10px 30px rgba(249,168,37,0.3);
}}

/* ============================================
   STADIUM GRAPHIC
   ============================================ */
.stadium-container {{
    text-align: center;
    padding: 2rem;
    animation: slideInUp 0.8s ease-out 0.6s both;
}}

.stadium-graphic {{
    display: inline-block;
    position: relative;
    width: 200px;
    height: 200px;
    animation: floatUp 4s ease-in-out infinite;
}}

.stadium-field {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 160px;
    height: 160px;
    background: radial-gradient(circle, #1b4d1b 0%, #0d260d 70%);
    border: 3px solid #f9a825;
    border-radius: 50%;
    animation: pulseGlow 3s ease-in-out infinite;
}}

.stadium-track {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 180px;
    height: 180px;
    border: 2px dashed rgba(249,168,37,0.5);
    border-radius: 50%;
    animation: rotate360 20s linear infinite;
}}

.stadium-spot {{
    position: absolute;
    width: 10px;
    height: 10px;
    background: #f9a825;
    border-radius: 50%;
    box-shadow: 0 0 15px #f9a825;
    animation: blinkLights 1.5s ease-in-out infinite;
}}

.stadium-spot.one {{ top: 20%; left: 20%; }}
.stadium-spot.two {{ top: 20%; right: 20%; animation-delay: 0.5s; }}
.stadium-spot.three {{ bottom: 20%; left: 50%; transform: translateX(-50%); animation-delay: 1s; }}
.stadium-spot.four {{ bottom: 20%; right: 20%; animation-delay: 1.5s; }}
.stadium-spot.five {{ top: 50%; left: 50%; transform: translate(-50%, -50%); animation-delay: 0.3s; }}

/* ============================================
   TEAM DETAILS SECTION
   ============================================ */
.team-section {{
    background: linear-gradient(145deg, #13131a, #18181f);
    border: 1px solid #2a2a3a;
    border-radius: 25px;
    padding: 3rem 2rem;
    margin: 3rem 0;
    position: relative;
    overflow: hidden;
    animation: slideInUp 1s ease-out;
}}

.team-section::before {{
    content: '';
    position: absolute;
    top: -100px;
    right: -100px;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(249,168,37,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}}

.team-section::after {{
    content: '';
    position: absolute;
    bottom: -100px;
    left: -100px;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(21,101,192,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite 2s;
}}

@keyframes pulse {{
    0%, 100% {{ transform: scale(1); opacity: 0.5; }}
    50% {{ transform: scale(1.1); opacity: 0.8; }}
}}

.team-header {{
    text-align: center;
    margin-bottom: 3rem;
    position: relative;
    z-index: 1;
}}

.team-header h2 {{
    font-size: 2.5rem;
    background: linear-gradient(135deg, #f9a825, #ffd54f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
}}

.team-subtitle {{
    color: #666;
    font-size: 0.95rem;
    letter-spacing: 1px;
}}

.team-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    position: relative;
    z-index: 1;
}}

.team-member {{
    background: linear-gradient(145deg, #1a1a24, #141418);
    border: 1px solid #2a2a3a;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}}

.team-member::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(249,168,37,0.1), transparent);
    transition: left 0.6s;
}}

.team-member:hover {{
    border-color: #f9a825;
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 20px 50px rgba(249,168,37,0.3);
}}

.team-member:hover::before {{
    left: 100%;
}}

.member-avatar {{
    width: 120px;
    height: 120px;
    margin: 0 auto 1.5rem;
    border-radius: 50%;
    background: linear-gradient(135deg, #f9a825, #ff6f00);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    color: #000;
    position: relative;
    animation: floatUp 3s ease-in-out infinite;
    box-shadow: 0 10px 30px rgba(249,168,37,0.4);
}}

.member-avatar::after {{
    content: '';
    position: absolute;
    inset: -5px;
    border-radius: 50%;
    border: 2px solid #f9a825;
    animation: spin 10s linear infinite;
    opacity: 0.3;
}}

.member-name {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f9a825;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
}}

.member-role {{
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.member-bio {{
    color: #aaa;
    font-size: 0.85rem;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}}

.member-links {{
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1rem;
}}

.member-links a {{
    color: #f9a825;
    text-decoration: none;
    font-size: 0.85rem;
    transition: color 0.3s ease;
}}

.member-links a:hover {{
    color: #fff;
}}

.member-skills {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
}}

.skill-badge {{
    background: rgba(249,168,37,0.15);
    border: 1px solid rgba(249,168,37,0.3);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-size: 0.75rem;
    color: #f9a825;
    transition: all 0.3s ease;
}}

.skill-badge:hover {{
    background: rgba(249,168,37,0.25);
    border-color: #f9a825;
    transform: scale(1.05);
}}

/* ============================================
   MAIN CONTENT LAYOUT
   ============================================ */
.main-container {{
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 2rem;
    margin-top: 1rem;
    animation: slideInUp 1s ease-out 0.8s both;
}}

/* ============================================
   FORM CARD
   ============================================ */
.form-card {{
    background: linear-gradient(145deg, #13131a, #18181f);
    border: 1px solid #2a2a3a;
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}}

.form-card::after {{
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 150px;
    height: 150px;
    background: radial-gradient(circle, rgba(249,168,37,0.1) 0%, transparent 70%);
    pointer-events: none;
}}

.form-card:hover {{
    border-color: #3a3a4a;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}}

.form-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #2a2a3a;
}}

.form-header i {{
    font-size: 1.5rem;
    color: #f9a825;
    animation: wave 2s ease-in-out infinite;
}}

.form-header h3 {{
    font-size: 1.4rem;
    color: #fff;
    letter-spacing: 1px;
}}

.form-label {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #aaa;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}}

.form-label i {{
    color: #f9a825;
    font-size: 0.9rem;
}}

.form-label .team1-color {{ color: #4fc3f7; }}
.form-label .team2-color {{ color: #ef5350; }}

/* Select box styling */
div[data-baseweb="select"] > div {{
    background: #1a1a24 !important;
    border: 2px solid #2a2a3a !important;
    border-radius: 10px !important;
    color: #fff !important;
    transition: all 0.3s ease !important;
}}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {{
    border-color: #f9a825 !important;
    box-shadow: 0 0 15px rgba(249,168,37,0.2) !important;
}}

.row-split {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}}

/* ============================================
   PREDICT BUTTON
   ============================================ */
.predict-btn {{
    margin-top: 1.5rem;
    position: relative;
}}

.predict-btn button {{
    background: linear-gradient(135deg, #f9a825, #ff6f00) !important;
    color: #000 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 2rem !important;
    width: 100% !important;
    letter-spacing: 2px !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}}

.predict-btn button::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.3),
        transparent
    );
    transition: left 0.5s;
}}

.predict-btn button:hover {{
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 15px 40px rgba(249,168,37,0.5) !important;
}}

.predict-btn button:hover::before {{
    left: 100%;
}}

/* ============================================
   RESULT SECTION
   ============================================ */
.result-section {{
    position: relative;
}}

.winner-card {{
    background: linear-gradient(145deg, #1a1a00, #252500);
    border: 2px solid #f9a825;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: scaleIn 0.5s ease-out, pulseGlow 3s ease-in-out 0.5s infinite;
    position: relative;
    overflow: hidden;
}}

.winner-card::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(249,168,37,0.1) 0%, transparent 50%);
    animation: rotate360 10s linear infinite;
}}

.winner-label {{
    color: #888;
    font-size: 0.9rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
}}

.winner-team-name {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #f9a825;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 30px rgba(249,168,37,0.5);
}}

.winner-team-name i {{
    display: inline-block;
    animation: bounce2 1.5s ease-in-out infinite;
    margin-right: 10px;
}}

.confidence-text {{
    color: #aaa;
    font-size: 1rem;
    margin-top: 1rem;
    position: relative;
    z-index: 1;
}}

.confidence-text strong {{
    color: #f9a825;
    font-size: 1.3rem;
    font-weight: 700;
}}

/* ============================================
   PROBABILITY BARS
   ============================================ */
.probability-chart {{
    margin: 1.5rem 0;
    animation: slideInUp 0.6s ease-out 0.3s both;
}}

.probability-cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
    animation: slideInUp 0.6s ease-out 0.4s both;
}}

.prob-card {{
    background: linear-gradient(145deg, #141418, #1a1a22);
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}}

.prob-card:hover {{
    border-color: #f9a825;
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(249,168,37,0.2);
}}

.prob-card .tag {{
    color: #666;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}}

.prob-card .value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f9a825;
}}

.prob-card .sub {{
    color: #444;
    font-size: 0.7rem;
    margin-top: 0.3rem;
}}

/* ============================================
   ANALYTICS TABS
   ============================================ */
.analytics-section {{
    margin-top: 2rem;
    animation: slideInUp 1s ease-out 1s both;
}}

div[data-testid="stTabbedContent"] {{
    background: linear-gradient(145deg, #13131a, #18181f);
    border: 1px solid #2a2a3a;
    border-radius: 20px;
    overflow: hidden;
}}

div[data-testid="stTabBar"] {{
    background: linear-gradient(90deg, #1a1a24, #13131a);
}}

div[data-testid="stExpander"] {{
    background: linear-gradient(145deg, #13131a, #18181f);
    border: 1px solid #2a2a3a;
    border-radius: 15px;
}}

div[data-testid="metric-container"] {{
    background: linear-gradient(145deg, #141418, #1a1a22);
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1rem;
    transition: all 0.3s ease;
}}

div[data-testid="metric-container"]:hover {{
    border-color: #f9a825;
    transform: translateY(-3px);
}}


.empty-state {{
    background: linear-gradient(145deg, #13131a, #18181f);
    border: 1px solid #2a2a3a;
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}

.empty-state::before,
.empty-state::after {{
    content: '';
    position: absolute;
    width: 150px;
    height: 150px;
    border-radius: 50%;
    filter: blur(60px);
}}

.empty-state::before {{
    top: -50px;
    left: -50px;
    background: rgba(249,168,37,0.1);
}}

.empty-state::after {{
    bottom: -50px;
    right: -50px;
    background: rgba(21,101,192,0.1);
}}

.empty-icon {{
    font-size: 4rem;
    color: #333;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
    animation: floatUp 3s ease-in-out infinite;
}}

.empty-title {{
    color: #f9a825;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    position: relative;
    z-index: 1;
}}

.empty-desc {{
    color: #777;
    font-size: 0.95rem;
    line-height: 1.7;
    position: relative;
    z-index: 1;
}}

.empty-desc strong {{
    color: #f9a825;
}}

/* ============================================
   FOOTER
   ============================================ */
.footer-section {{
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid #1e1e2a;
    animation: fadeInDelay 1s ease-out 1.2s both;
}}

.footer-icons {{
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 1rem;
}}

.footer-icons a {{
    color: #555;
    font-size: 1.5rem;
    transition: all 0.3s ease;
}}

.footer-icons a:hover {{
    color: #f9a825;
    transform: translateY(-5px);
}}

.footer-text {{
    color: #444;
    font-size: 0.85rem;
}}

.footer-text span {{
    color: #f9a825;
}}

/* ============================================
   RESPONSIVE
   ============================================ */
@media (max-width: 1024px) {{
    .stats-container {{
        grid-template-columns: repeat(3, 1fr);
    }}
    .main-container {{
        grid-template-columns: 1fr;
    }}
    .probability-cards {{
        grid-template-columns: 1fr;
    }}
    .team-grid {{
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    }}
}}

@media (max-width: 768px) {{
    .stats-container {{
        grid-template-columns: repeat(2, 1fr);
    }}
    .hero-title {{
        font-size: 2.5rem;
    }}
    .feature-pills {{
        flex-direction: column;
        align-items: center;
    }}
    .team-grid {{
        grid-template-columns: 1fr;
    }}
}}

/* Custom scrollbar */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: #0a0a0f;
}}

::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg, #f9a825, #ff6f00);
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: linear-gradient(180deg, #ff6f00, #f9a825);
}}
</style>
""", unsafe_allow_html=True)

# Load the IPL match dataset, preprocess it, and train the prediction model
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("matches.csv")
    except FileNotFoundError:
        st.error("matches.csv not found. Please download from Kaggle.")
        st.code("https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set")
        st.stop()

    df = df.dropna(subset=['winner'])
    df['toss_win_match_win'] = (df['toss_winner'] == df['winner']).astype(int)

    features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
    target = 'winner'
    df_model = df[features + [target]].dropna()

    encoders = {}
    df_enc = df_model.copy()
    for col in features + [target]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc[features]
    y = df_enc[target]
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X, y)
    
    accuracy = cross_val_score(model, X, y, cv=5).mean() * 100

    teams = sorted(set(df['team1'].dropna().tolist() + df['team2'].dropna().tolist()))
    venues = sorted(df['venue'].dropna().unique().tolist())

    stats = {
        'total_matches': len(df),
        'seasons': df['season'].nunique() if 'season' in df.columns else 'N/A',
        'teams': len(teams),
        'toss_impact': round(df['toss_win_match_win'].mean() * 100, 1),
        'accuracy': round(accuracy, 1)
    }

    return model, encoders, teams, venues, df, stats

model, encoders, teams, venues, df, stats = load_and_train()

# Render the hero header section at the top of the page
st.markdown("""
<div class="hero-section">
    <div class="hero-icon">
        <i class="fas fa-cricket"></i>
    </div>
    <h1 class="hero-title">IPL WIN PREDICTOR</h1>
    <p class="hero-subtitle">Machine Learning &bull; Random Forest &bull; Scikit-Learn</p>
</div>
""", unsafe_allow_html=True)

# Display a row of key IPL stats for quick context
st.markdown(f"""
<div class="stats-container">
    <div class="stat-card">
        <i class="fas fa-list-alt stat-icon"></i>
        <div class="stat-value">{stats['total_matches']}</div>
        <div class="stat-label">Total Matches</div>
    </div>
    <div class="stat-card">
        <i class="fas fa-calendar-alt stat-icon"></i>
        <div class="stat-value">{stats['seasons']}</div>
        <div class="stat-label">IPL Seasons</div>
    </div>
    <div class="stat-card">
        <i class="fas fa-users stat-icon"></i>
        <div class="stat-value">{stats['teams']}</div>
        <div class="stat-label">Teams</div>
    </div>
    <div class="stat-card">
        <i class="fas fa-coins stat-icon"></i>
        <div class="stat-value">{stats['toss_impact']}%</div>
        <div class="stat-label">Toss Impact</div>
    </div>
    <div class="stat-card">
        <i class="fas fa-crosshairs stat-icon"></i>
        <div class="stat-value">{stats['accuracy']}%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Add a small row of feature highlights with icons
st.markdown("""
<div class="feature-pills">
    <div class="feature-pill">
        <i class="fas fa-bolt"></i>
        Venue-aware prediction
    </div>
    <div class="feature-pill">
        <i class="fas fa-chart-line"></i>
        Data-driven insights
    </div>
    <div class="feature-pill">
        <i class="fas fa-brain"></i>
        ML-powered forecasting
    </div>
    <div class="feature-pill">
        <i class="fas fa-shield-alt"></i>
        Toss intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# Render the stadium-style graphic section for visual polish
st.markdown("""
<div class="stadium-container">
    <div class="stadium-graphic">
        <div class="stadium-track"></div>
        <div class="stadium-field"></div>
        <div class="stadium-spot one"></div>
        <div class="stadium-spot two"></div>
        <div class="stadium-spot three"></div>
        <div class="stadium-spot four"></div>
        <div class="stadium-spot five"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Display the project team cards and contributor bios
st.markdown("""
<div class="team-section">
    <div class="team-header">
        <h2><i class="fas fa-users-cog"></i> Meet Our Team</h2>
        <p class="team-subtitle">The Minds Behind This IPL Prediction System</p>
    </div>
</div>
""", unsafe_allow_html=True)

team_cols = st.columns(3, gap="large")

with team_cols[0]:
    st.markdown("""
    <div class="team-member">
        <div class="member-avatar">
            <i class="fas fa-user-tie"></i>
        </div>
        <div class="member-name">Suryansh Pandey</div>
        <div class="member-role">Project Lead & ML Engineer</div>
        <div class="member-bio">
            Led the development of the prediction model using Random Forest algorithms and coordinated the entire project workflow.
        </div>
        <div class="member-skills">
            <span class="skill-badge">Machine Learning</span>
            <span class="skill-badge">Python</span>
            <span class="skill-badge">Scikit-Learn</span>
            <span class="skill-badge">Data Analysis</span>
            <span class="skill-badge">Project Management</span>
        </div>
        <div class="member-links">
            <a href="https://www.linkedin.com/in/suryansh-pandey-253635329" target="_blank"><i class="fab fa-linkedin"></i> LinkedIn</a>
            <a href="mailto:rbharadwaj364@gmail.com"><i class="fas fa-envelope"></i> Email</a>
            <a href="https://github.com/rishigit04" target="_blank"><i class="fab fa-github"></i> GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with team_cols[1]:
    st.markdown("""
    <div class="team-member">
        <div class="member-avatar">
            <i class="fas fa-palette"></i>
        </div>
        <div class="member-name">Sidharth Singh</div>
        <div class="member-role">UI/UX Designer</div>
        <div class="member-bio">
            Designed the interactive web interface with advanced visualizations and created an intuitive user experience.
        </div>
        <div class="member-skills">
            <span class="skill-badge">UI/UX Design</span>
            <span class="skill-badge">CSS/HTML</span>
            <span class="skill-badge">Streamlit</span>
            <span class="skill-badge">Plotly</span>
            <span class="skill-badge">Responsive Design</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with team_cols[2]:
    st.markdown("""
    <div class="team-member">
        <div class="member-avatar">
            <i class="fas fa-code"></i>
        </div>
        <div class="member-name">Naman Kumar Mishra</div>
        <div class="member-role">Python Developer</div>
        <div class="member-bio">
            Implemented core Python functionality, data preprocessing, and integrated machine learning models with the frontend.
        </div>
        <div class="member-skills">
            <span class="skill-badge">Python</span>
            <span class="skill-badge">Pandas</span>
            <span class="skill-badge">NumPy</span>
            <span class="skill-badge">Data Engineering</span>
            <span class="skill-badge">Backend Development</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render the main app content layout with input controls on the left
left, right = st.columns([1, 1.2], gap="large")

with left:
    # Build the match input form panel
    st.markdown("""
    <div class="form-card">
        <div class="form-header">
            <i class="fas fa-cogs"></i>
            <h3>Match Setup</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="form-label"><i class="fas fa-circle team1-color"></i> Team 1</div>', unsafe_allow_html=True)
    team1 = st.selectbox("Select Team 1", teams, index=0, key="team1_select", label_visibility="collapsed")
    
    st.markdown('<div class="form-label"><i class="fas fa-circle team2-color"></i> Team 2</div>', unsafe_allow_html=True)
    team2_options = [t for t in teams if t != team1]
    team2 = st.selectbox("Select Team 2", team2_options, index=0, key="team2_select", label_visibility="collapsed")
    
    st.markdown('<div class="form-label"><i class="fas fa-map-marker-alt"></i> Venue</div>', unsafe_allow_html=True)
    venue = st.selectbox("Select Venue", venues, key="venue_select", label_visibility="collapsed")
    
    st.markdown('<div class="row-split">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="form-label"><i class="fas fa-trophy"></i> Toss Winner</div>', unsafe_allow_html=True)
        toss_winner = st.selectbox("Toss Winner", [team1, team2], key="toss_winner_select", label_visibility="collapsed")
    
    with col_b:
        st.markdown('<div class="form-label"><i class="fas fa-clipboard-list"></i> Decision</div>', unsafe_allow_html=True)
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"], key="toss_decision_select", label_visibility="collapsed")
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict = st.button("PREDICT WINNER")
    st.markdown('</div>', unsafe_allow_html=True)

# Show the prediction results and analytics panel on the right side
with right:
    if predict:
        with st.spinner('Analyzing match data...'):
            def safe_encode(encoder, value):
                classes = list(encoder.classes_)
                if value not in classes:
                    return 0
                return encoder.transform([value])[0]

            input_data = pd.DataFrame([{
                'team1': safe_encode(encoders['team1'], team1),
                'team2': safe_encode(encoders['team2'], team2),
                'toss_winner': safe_encode(encoders['toss_winner'], toss_winner),
                'toss_decision': safe_encode(encoders['toss_decision'], toss_decision),
                'venue': safe_encode(encoders['venue'], venue),
            }])

            proba = model.predict_proba(input_data)[0]
            pred_class = model.predict(input_data)[0]
            winner = encoders['winner'].inverse_transform([pred_class])[0]

            winner_classes = list(encoders['winner'].classes_)
            t1_idx = winner_classes.index(team1) if team1 in winner_classes else None
            t2_idx = winner_classes.index(team2) if team2 in winner_classes else None
            t1_prob = round(proba[t1_idx] * 100, 1) if t1_idx is not None else 50.0
            t2_prob = round(proba[t2_idx] * 100, 1) if t2_idx is not None else 50.0
            confidence = max(t1_prob, t2_prob)

        # Show the predicted winner banner
        st.markdown(f"""
        <div class="winner-card">
            <div class="winner-label"><i class="fas fa-bullseye"></i> Predicted Winner</div>
            <div class="winner-team-name">
                <i class="fas fa-trophy"></i>{winner}
            </div>
            <div class="confidence-text">
                Confidence: <strong>{confidence}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability Bar Chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=['Win Probability'],
            x=[t1_prob],
            name=team1,
            orientation='h',
            marker=dict(color='#f9a825', line=dict(color='#ff6f00', width=2)),
            text=f'{team1}<br>{t1_prob}%',
            textposition='inside',
            textfont=dict(size=13, color='#000', family='Rajdhani', weight='bold'),
            hovertemplate=f'<b>{team1}</b><br>Win Probability: {t1_prob}%<extra></extra>'
        ))
        fig_bar.add_trace(go.Bar(
            y=['Win Probability'],
            x=[t2_prob],
            name=team2,
            orientation='h',
            marker=dict(color='#1565c0', line=dict(color='#0d47a1', width=2)),
            text=f'{team2}<br>{t2_prob}%',
            textposition='inside',
            textfont=dict(size=13, color='#fff', family='Rajdhani', weight='bold'),
            hovertemplate=f'<b>{team2}</b><br>Win Probability: {t2_prob}%<extra></extra>'
        ))
        fig_bar.update_layout(
            barmode='stack',
            plot_bgcolor='#13131a',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            height=100,
            showlegend=False,
            xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.markdown('<div class="probability-chart">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Probability Cards
        st.markdown(f"""
        <div class="probability-cards">
            <div class="prob-card">
                <div class="tag"><i class="fas fa-shield-alt"></i> {team1}</div>
                <div class="value">{t1_prob}%</div>
                <div class="sub">Win Probability</div>
            </div>
            <div class="prob-card">
                <div class="tag"><i class="fas fa-shield-alt"></i> {team2}</div>
                <div class="value">{t2_prob}%</div>
                <div class="sub">Win Probability</div>
            </div>
            <div class="prob-card">
                <div class="tag"><i class="fas fa-coins"></i> Toss Impact</div>
                <div class="value">{stats['toss_impact']}%</div>
                <div class="sub">Historical</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Donut Chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=[team1, team2],
            values=[t1_prob, t2_prob],
            hole=0.6,
            marker=dict(colors=['#f9a825', '#1565c0'], line=dict(color='#0a0a0f', width=3)),
            textinfo='label+percent',
            textfont=dict(size=12, family='Poppins', color='#fff'),
            hovertemplate='<b>%{label}</b><br>Win Probability: %{value}%<extra></extra>'
        )])
        fig_donut.add_annotation(
            text=f'<b>{confidence}%</b>',
            x=0.5, y=0.5,
            font=dict(size=22, color='#f9a825', family='Rajdhani'),
            showarrow=False
        )
        fig_donut.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=280,
            showlegend=True,
            legend=dict(
                orientation='h', 
                yanchor='bottom', 
                y=-0.1, 
                xanchor='center', 
                x=0.5,
                font=dict(color='#e0e0e0')
            ),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Head to Head Stats
        st.markdown("### <i class='fas fa-users'></i> Head-to-Head", unsafe_allow_html=True)
        h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) |
                  ((df['team1'] == team2) & (df['team2'] == team1))]
        
        if len(h2h) > 0:
            t1_wins = len(h2h[h2h['winner'] == team1])
            t2_wins = len(h2h[h2h['winner'] == team2])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Matches", len(h2h))
            c2.metric(f"{team1}", t1_wins)
            c3.metric(f"{team2}", t2_wins)
            
            fig_h2h = go.Figure(data=[
                go.Bar(
                    x=[team1, team2], 
                    y=[t1_wins, t2_wins],
                    marker=dict(
                        color=['#f9a825', '#1565c0'],
                        line=dict(color=['#ff6f00', '#0d47a1'], width=2)
                    ),
                    text=[t1_wins, t2_wins], 
                    textposition='outside',
                    textfont=dict(size=14, family='Rajdhani', color='#e0e0e0'),
                    hovertemplate='<b>%{x}</b><br>Wins: %{y}<extra></extra>'
                )
            ])
            fig_h2h.update_layout(
                plot_bgcolor='#13131a', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0'), 
                height=250,
                title=dict(text='Historical Wins', font=dict(size=14, family='Rajdhani')),
                xaxis=dict(showgrid=False, showline=True, linecolor='#2a2a3a'),
                yaxis=dict(showgrid=True, gridcolor='#222', showline=True, linecolor='#2a2a3a'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_h2h, use_container_width=True)
        else:
            st.info("No head-to-head data available")

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">
                <i class="fas fa-cricket"></i>
            </div>
            <div class="empty-title">Ready to Predict?</div>
            <div class="empty-desc">
                Select your teams, venue, and toss details,<br>
                then click <strong>Predict Winner</strong> to see the forecast.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Add analytics tabs so users can explore deeper model insights
st.markdown("---")
st.markdown("### <i class='fas fa-chart-pie'></i> Advanced Analytics", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Feature Importance",
    "Team Performance",
    "Toss Analysis",
    "Venue Insights"
])

with tab1:
    feature_names = ['Team 1', 'Team 2', 'Toss Winner', 'Toss Decision', 'Venue']
    importances = model.feature_importances_
    
    fig_imp = go.Figure(data=[
        go.Bar(
            x=importances, 
            y=feature_names, 
            orientation='h',
            marker=dict(
                color=importances, 
                colorscale='YlOrRd',
                line=dict(color='#f9a825', width=1)
            ),
            text=[f'{imp:.4f}' for imp in importances],
            textposition='outside',
            textfont=dict(size=11, family='Rajdhani', color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    ])
    fig_imp.update_layout(
        plot_bgcolor='#13131a', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'), 
        height=350,
        title=dict(text='Model Feature Importance', font=dict(size=16, family='Rajdhani')),
        xaxis=dict(title='Importance Score', showgrid=True, gridcolor='#222', showline=True, linecolor='#2a2a3a'),
        yaxis=dict(showgrid=False, showline=True, linecolor='#2a2a3a'),
        margin=dict(l=20, r=80, t=40, b=20)
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("<i class='fas fa-info-circle'></i> Venue and team identity are the strongest predictors of match outcome.", unsafe_allow_html=True)

with tab2:
    all_teams = sorted(set(df['team1'].tolist() + df['team2'].tolist()))
    team_stats = []
    for team in all_teams:
        matches = len(df[(df['team1'] == team) | (df['team2'] == team)])
        wins = len(df[df['winner'] == team])
        win_rate = (wins / matches * 100) if matches > 0 else 0
        team_stats.append({'Team': team, 'Matches': matches, 'Wins': wins, 'Win Rate': round(win_rate, 1)})
    
    team_df = pd.DataFrame(team_stats).sort_values('Win Rate', ascending=False)
    
    fig_teams = go.Figure(data=[
        go.Bar(
            x=team_df['Team'], 
            y=team_df['Win Rate'],
            marker=dict(
                color=team_df['Win Rate'], 
                colorscale='YlOrRd',
                line=dict(color='#f9a825', width=1)
            ),
            text=team_df['Win Rate'].apply(lambda x: f'{x}%'),
            textposition='outside', 
            textfont=dict(size=9, family='Rajdhani', color='#e0e0e0'),
            hovertemplate='<b>%{x}</b><br>Win Rate: %{y}%<extra></extra>'
        )
    ])
    fig_teams.update_layout(
        plot_bgcolor='#13131a', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'), 
        height=450,
        title=dict(text='Team Win Rates', font=dict(size=16, family='Rajdhani')),
        xaxis=dict(tickangle=-45, showgrid=False, showline=True, linecolor='#2a2a3a'),
        yaxis=dict(title='Win Rate (%)', showgrid=True, gridcolor='#222', range=[0, 70], showline=True, linecolor='#2a2a3a'),
        margin=dict(l=20, r=20, t=40, b=100)
    )
    st.plotly_chart(fig_teams, use_container_width=True)
    st.dataframe(team_df.style.background_gradient(cmap='YlOrRd', subset=['Win Rate']), use_container_width=True, height=350)

with tab3:
    toss_decision_counts = df['toss_decision'].value_counts()
    fig_toss = go.Figure(data=[
        go.Pie(
            labels=toss_decision_counts.index, 
            values=toss_decision_counts.values,
            hole=0.4, 
            marker=dict(
                colors=['#f9a825', '#1565c0'],
                line=dict(color='#0a0a0f', width=2)
            ),
            textinfo='label+percent+value',
            textfont=dict(size=12, family='Poppins', color='#fff'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    fig_toss.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        height=350,
        title=dict(text='Toss Decision Distribution', font=dict(size=16, family='Rajdhani', color='#e0e0e0')),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_toss, use_container_width=True)
    
    bat_wins = len(df[(df['toss_decision'] == 'bat') & (df['toss_winner'] == df['winner'])])
    field_wins = len(df[(df['toss_decision'] == 'field') & (df['toss_winner'] == df['winner'])])
    bat_total = len(df[df['toss_decision'] == 'bat'])
    field_total = len(df[df['toss_decision'] == 'field'])
    bat_rate = (bat_wins / bat_total * 100) if bat_total > 0 else 0
    field_rate = (field_wins / field_total * 100) if field_total > 0 else 0
    
    fig_impact = go.Figure(data=[
        go.Bar(
            x=['Bat First', 'Field First'], 
            y=[bat_rate, field_rate],
            marker=dict(
                color=['#f9a825', '#1565c0'],
                line=dict(color=['#ff6f00', '#0d47a1'], width=2)
            ),
            text=[f'{bat_rate:.1f}%', f'{field_rate:.1f}%'], 
            textposition='outside',
            textfont=dict(size=13, family='Rajdhani', color='#e0e0e0'),
            hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>'
        )
    ])
    fig_impact.update_layout(
        plot_bgcolor='#13131a', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'), 
        height=300,
        title=dict(text='Win Rate by Toss Decision', font=dict(size=16, family='Rajdhani')),
        yaxis=dict(range=[0, 60], showgrid=True, gridcolor='#222', showline=True, linecolor='#2a2a3a'),
        xaxis=dict(showgrid=False, showline=True, linecolor='#2a2a3a'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_impact, use_container_width=True)

with tab4:
    venue_counts = df['venue'].value_counts().head(15)
    fig_venues = go.Figure(data=[
        go.Bar(
            x=venue_counts.index, 
            y=venue_counts.values,
            marker=dict(
                color=venue_counts.values, 
                colorscale='YlOrRd',
                line=dict(color='#f9a825', width=1)
            ),
            text=venue_counts.values, 
            textposition='outside',
            textfont=dict(size=10, family='Rajdhani', color='#e0e0e0'),
            hovertemplate='<b>%{x}</b><br>Total Matches: %{y}<extra></extra>'
        )
    ])
    fig_venues.update_layout(
        plot_bgcolor='#13131a', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'), 
        height=450,
        title=dict(text='Top 15 Venues', font=dict(size=16, family='Rajdhani')),
        xaxis=dict(tickangle=-45, showgrid=False, showline=True, linecolor='#2a2a3a'),
        yaxis=dict(title='Matches', showgrid=True, gridcolor='#222', showline=True, linecolor='#2a2a3a'),
        margin=dict(l=20, r=20, t=40, b=150)
    )
    st.plotly_chart(fig_venues, use_container_width=True)

# Add the footer section with final notes and credits
st.markdown("""
<div class="footer-section">
    <div class="footer-icons">
        <a href="#"><i class="fab fa-python"></i></a>
        <a href="#"><i class="fas fa-brain"></i></a>
        <a href="#"><i class="fas fa-chart-bar"></i></a>
        <a href="#"><i class="fas fa-database"></i></a>
    </div>
    <div class="footer-text">
        Built with <span><i class="fas fa-heart"></i></span> by Suryansh, Sidharth & Naman
    </div>
    <div style="color:#333;font-size:0.75rem;margin-top:0.5rem">
        IPL Win Predictor &bull; Data Science Project &bull; 2024
    </div>
</div>
""", unsafe_allow_html=True)