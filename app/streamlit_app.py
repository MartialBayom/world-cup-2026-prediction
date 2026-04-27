# ─────────────────────────────────────────────────────
# app/streamlit_app.py
# Dashboard CDM 2026 — Prédictions ML
# ─────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import boto3
import joblib
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

load_dotenv(".env")

st.set_page_config(
    page_title="CDM 2026 — Prédictions ML",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0D1B2A; }
    [data-testid="stSidebar"] { background-color: #1A3A5C; }
    [data-testid="stSidebar"] * { color: white !important; }
    h1, h2, h3 { color: #2DD4BF !important; }
    [data-testid="stMetric"] {
        background-color: #1A3A5C;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2DD4BF;
    }
    [data-testid="stMetricValue"] { color: #2DD4BF !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: white !important; }
    .stButton > button[kind="primary"] {
        background-color: #2DD4BF !important;
        color: #0D1B2A !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton > button[kind="primary"]:hover { background-color: #14B8A6 !important; }
    hr { border-color: #2DD4BF33 !important; }
    .card {
        background-color: #1A3A5C;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #2DD4BF44;
        margin-bottom: 6px;
    }
    .card-winner {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

FLAGS = {
    'Algeria': '🇩🇿', 'Argentina': '🇦🇷', 'Australia': '🇦🇺', 'Austria': '🇦🇹',
    'Belgium': '🇧🇪', 'Bosnia and Herzegovina': '🇧🇦', 'Brazil': '🇧🇷',
    'Canada': '🇨🇦', 'Cape Verde': '🇨🇻', 'Colombia': '🇨🇴', 'Croatia': '🇭🇷',
    'Czech Republic': '🇨🇿', 'DR Congo': '🇨🇩', 'Ecuador': '🇪🇨', 'Egypt': '🇪🇬',
    'England': '🏴', 'France': '🇫🇷', 'Germany': '🇩🇪', 'Ghana': '🇬🇭',
    'Haiti': '🇭🇹', 'Iran': '🇮🇷', 'Iraq': '🇮🇶', 'Ivory Coast': '🇨🇮',
    'Japan': '🇯🇵', 'Jordan': '🇯🇴', 'Mexico': '🇲🇽', 'Morocco': '🇲🇦',
    'Netherlands': '🇳🇱', 'New Zealand': '🇳🇿', 'Norway': '🇳🇴', 'Panama': '🇵🇦',
    'Paraguay': '🇵🇾', 'Portugal': '🇵🇹', 'Qatar': '🇶🇦', 'Saudi Arabia': '🇸🇦',
    'Scotland': '🏴', 'Senegal': '🇸🇳', 'South Africa': '🇿🇦',
    'South Korea': '🇰🇷', 'Spain': '🇪🇸', 'Sweden': '🇸🇪', 'Switzerland': '🇨🇭',
    'Tunisia': '🇹🇳', 'Turkey': '🇹🇷', 'United States': '🇺🇸', 'Uruguay': '🇺🇾',
    'Uzbekistan': '🇺🇿', 'Curacao': '🇨🇼',
}

def flag(team):
    return FLAGS.get(team, '🏳️')

BUCKET  = os.getenv("S3_BUCKET")
STORAGE = {
    "key"           : os.getenv("AWS_ACCESS_KEY_ID"),
    "secret"        : os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs" : {"region_name": os.getenv("AWS_REGION")}
}

@st.cache_resource
def load_model_s3(model_name):
    s3  = boto3.client('s3',
        aws_access_key_id     = os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name           = os.getenv('AWS_REGION'))
    obj = s3.get_object(Bucket=BUCKET, Key="models/" + model_name + ".pkl")
    return joblib.load(io.BytesIO(obj['Body'].read()))

@st.cache_data
def load_data():
    PROC       = "s3://" + BUCKET + "/processed"
    team_stats = pd.read_csv(PROC + "/team_stats_raw.csv",   storage_options=STORAGE)
    features   = pd.read_csv(PROC + "/dataset_features.csv", storage_options=STORAGE)
    return team_stats, features

GROUPES = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

ALL_TEAMS = sorted(set([t for teams in GROUPES.values() for t in teams]))

def get_team_stats(team, team_stats_df):
    past  = team_stats_df[team_stats_df['team'] == team].sort_values('date')
    last5 = past.tail(5)
    if len(past) == 0:
        return None
    return {
        'rank'        : past['rank'].values[-1],
        'goals'       : past['score'].mean(),
        'goals_l5'    : last5['score'].mean(),
        'goals_suf'   : past['suf_score'].mean(),
        'goals_suf_l5': last5['suf_score'].mean(),
        'rank_suf'    : past['rank_suf'].mean(),
        'rank_suf_l5' : last5['rank_suf'].mean(),
        'gp_rank'     : past['points_by_rank'].mean(),
        'gp_rank_l5'  : last5['points_by_rank'].mean(),
    }

def build_match_features(home_team, away_team, team_stats_df):
    t1 = get_team_stats(home_team, team_stats_df)
    t2 = get_team_stats(away_team, team_stats_df)
    if t1 is None or t2 is None:
        return None
    return pd.DataFrame([{
        'rank_dif'             : t1['rank'] - t2['rank'],
        'goals_dif'            : t1['goals'] - t2['goals'],
        'goals_dif_l5'         : t1['goals_l5'] - t2['goals_l5'],
        'goals_suf_dif'        : t1['goals_suf'] - t2['goals_suf'],
        'goals_suf_dif_l5'     : t1['goals_suf_l5'] - t2['goals_suf_l5'],
        'goals_per_ranking_dif': (t1['goals'] / t1['rank_suf']) - (t2['goals'] / t2['rank_suf']),
        'dif_rank_agst'        : t1['rank_suf'] - t2['rank_suf'],
        'dif_rank_agst_l5'     : t1['rank_suf_l5'] - t2['rank_suf_l5'],
        'dif_points_rank'      : t1['gp_rank'] - t2['gp_rank'],
        'dif_points_rank_l5'   : t1['gp_rank_l5'] - t2['gp_rank_l5'],
        'is_friendly'          : 0
    }])

def predict_match(home, away, model, team_stats_df, model_name, features_df):
    X     = build_match_features(home, away, team_stats_df)
    X_inv = build_match_features(away, home, team_stats_df)
    if X is None or X_inv is None:
        return None
    if model_name == "logistic_regression":
        scaler = StandardScaler()
        FEAT   = list(X.columns)
        scaler.fit(features_df[FEAT])
        X_arr     = scaler.transform(X)
        X_inv_arr = scaler.transform(X_inv)
    else:
        X_arr     = X.values
        X_inv_arr = X_inv.values
    p1     = model.predict_proba(X_arr)[0]
    p2     = model.predict_proba(X_inv_arr)[0]
    p_home = (p1[0] + p2[1]) / 2
    p_away = (p2[0] + p1[1]) / 2
    return {
        'home_team': home,
        'away_team': away,
        'P_home'   : round(p_home * 100, 1),
        'P_away'   : round(p_away * 100, 1)
    }

def simuler_groupe(groupe, equipes, model, team_stats_df, model_name, features_df):
    points = {eq: 0 for eq in equipes}
    buts   = {eq: 0 for eq in equipes}
    probs  = {eq: [] for eq in equipes}
    for i in range(len(equipes)):
        for j in range(i + 1, len(equipes)):
            r = predict_match(equipes[i], equipes[j], model, team_stats_df, model_name, features_df)
            if r is None:
                continue
            probs[equipes[i]].append(r['P_home'])
            probs[equipes[j]].append(r['P_away'])
            if r['P_home'] > r['P_away'] + 10:
                points[equipes[i]] += 3
                buts[equipes[i]]   += 2
            elif r['P_away'] > r['P_home'] + 10:
                points[equipes[j]] += 3
                buts[equipes[j]]   += 2
            else:
                points[equipes[i]] += 1
                points[equipes[j]] += 1
                buts[equipes[i]]   += 1
                buts[equipes[j]]   += 1
    classement = sorted(
        equipes,
        key=lambda x: (points[x], buts[x], np.mean(probs[x]) if probs[x] else 0),
        reverse=True
    )
    troisieme = {
        'team'  : classement[2],
        'points': points[classement[2]],
        'buts'  : buts[classement[2]],
        'groupe': groupe
    }
    return classement[:2], troisieme

def simuler_tournoi(model, team_stats_df, model_name, features_df):
    qualifies_directs = []
    troisièmes        = []
    resultats_groupes = {}

    for groupe, equipes in GROUPES.items():
        top2, troisieme = simuler_groupe(groupe, equipes, model, team_stats_df, model_name, features_df)
        qualifies_directs.extend(top2)
        troisièmes.append(troisieme)
        resultats_groupes[groupe] = top2

    meilleurs_tiers = [
        t['team'] for t in sorted(
            troisièmes, key=lambda x: (x['points'], x['buts']), reverse=True
        )[:8]
    ]
    tous_qualifies = qualifies_directs + meilleurs_tiers

    def jouer_match(h, a):
        r = predict_match(h, a, model, team_stats_df, model_name, features_df)
        return h if r is None or r['P_home'] >= r['P_away'] else a

    rounds = {
        'Seizièmes'   : [],
        'Huitièmes'   : [],
        'Quarts'      : [],
        'Demi-finales': [],
        'Finale'      : []
    }

    equipes = tous_qualifies
    for round_name in ['Seizièmes', 'Huitièmes', 'Quarts', 'Demi-finales']:
        suivants = []
        for i in range(0, len(equipes), 2):
            if i + 1 < len(equipes):
                w = jouer_match(equipes[i], equipes[i + 1])
                rounds[round_name].append((equipes[i], equipes[i + 1], w))
                suivants.append(w)
        equipes = suivants

    vainqueur = jouer_match(equipes[0], equipes[1])
    rounds['Finale'].append((equipes[0], equipes[1], vainqueur))
    return rounds, vainqueur, resultats_groupes, meilleurs_tiers

# ── SIDEBAR ───────────────────────────────────────────
st.sidebar.markdown("## ⚽ CDM 2026")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🔮 Prédicteur", "🏆 Simulation"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Modèle ML")

MODEL_OPTIONS = {
    "GradientBoosting"   : "gradient_boosting",
    "XGBoost"            : "xgboost",
    "Random Forest"      : "random_forest",
    "Logistic Regression": "logistic_regression"
}
model_label = st.sidebar.selectbox("", list(MODEL_OPTIONS.keys()))
model_name  = MODEL_OPTIONS[model_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.75rem; color:#94A3B8;'>"
    "📊 49 215 matchs analysés<br>"
    "🎯 AUC-ROC : 0.774<br>"
    "✅ Accuracy : 73%"
    "</div>",
    unsafe_allow_html=True
)

with st.spinner("⏳ Chargement des données..."):
    team_stats, features_df = load_data()
    model = load_model_s3(model_name)

# ══════════════════════════════════════════════════════
# PAGE 1 — PRÉDICTEUR
# ══════════════════════════════════════════════════════
if page == "🔮 Prédicteur":

    st.markdown("# 🔮 Prédicteur de match")
    st.markdown("Modèle : **" + model_label + "** · AUC-ROC 0.774 · Accuracy 73%")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox(
            "⚽ Équipe 1",
            ALL_TEAMS,
            index=ALL_TEAMS.index("France"),
            format_func=lambda x: flag(x) + " " + x
        )
    with col2:
        away = st.selectbox(
            "⚽ Équipe 2",
            ALL_TEAMS,
            index=ALL_TEAMS.index("Brazil"),
            format_func=lambda x: flag(x) + " " + x
        )

    stats_home = get_team_stats(home, team_stats)
    stats_away = get_team_stats(away, team_stats)

    if stats_home and stats_away:
        st.markdown("#### 📊 Statistiques des équipes")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(flag(home) + " Rang FIFA", "#" + str(int(stats_home['rank'])))
        c2.metric("⚽ Buts/match", str(round(stats_home['goals'], 2)))
        c3.metric("🛡️ Concédés/match", str(round(stats_home['goals_suf'], 2)))
        c4.metric(flag(away) + " Rang FIFA", "#" + str(int(stats_away['rank'])))
        c5.metric("⚽ Buts/match", str(round(stats_away['goals'], 2)))
        c6.metric("🛡️ Concédés/match", str(round(stats_away['goals_suf'], 2)))

    st.divider()

    if st.button("🔮 Prédire le résultat", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            result = predict_match(home, away, model, team_stats, model_name, features_df)

        if result is None:
            st.error("Données insuffisantes pour ces équipes.")
        else:
            winner = home if result['P_home'] > result['P_away'] else away
            loser  = away if winner == home else home

            st.markdown("### Résultat")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    flag(home) + " " + home,
                    str(result['P_home']) + "%",
                    delta="Favori ✅" if result['P_home'] > result['P_away'] else None
                )
            with col2:
                st.metric(
                    flag(away) + " " + away,
                    str(result['P_away']) + "%",
                    delta="Favori ✅" if result['P_away'] > result['P_home'] else None
                )

            color_home = '#2DD4BF' if result['P_home'] > result['P_away'] else '#EF4444'
            color_away = '#2DD4BF' if result['P_away'] > result['P_home'] else '#EF4444'

            fig = go.Figure(go.Bar(
                x=[flag(home) + " " + home, flag(away) + " " + away],
                y=[result['P_home'], result['P_away']],
                marker_color=[color_home, color_away],
                text=[str(result['P_home']) + "%", str(result['P_away']) + "%"],
                textposition='outside',
                textfont=dict(size=18, color='white')
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,27,42,0.8)',
                font_color='white',
                title=dict(
                    text=flag(home) + " " + home + " vs " + flag(away) + " " + away,
                    font=dict(size=20, color='#2DD4BF')
                ),
                yaxis=dict(title="Probabilité de victoire (%)", range=[0, 110], gridcolor='#1A3A5C'),
                xaxis=dict(gridcolor='#1A3A5C'),
                showlegend=False,
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            winner_pct = max(result['P_home'], result['P_away'])
            st.markdown(
                "<div class='card-winner'>"
                "<h2 style='color:#0D1B2A; margin:0;'>"
                + flag(winner) + " " + winner + " est favori à " + str(winner_pct) + "%"
                + "</h2></div>",
                unsafe_allow_html=True
            )

            if stats_home and stats_away:
                st.markdown("#### 📈 Comparaison des statistiques")
                categories = ['Buts/match', 'Buts encaissés', 'Buts (5 derniers)', 'Enc. (5 derniers)']
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[stats_home['goals'], stats_home['goals_suf'],
                       stats_home['goals_l5'], stats_home['goals_suf_l5']],
                    theta=categories,
                    fill='toself',
                    name=flag(home) + " " + home,
                    line_color='#2DD4BF'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=[stats_away['goals'], stats_away['goals_suf'],
                       stats_away['goals_l5'], stats_away['goals_suf_l5']],
                    theta=categories,
                    fill='toself',
                    name=flag(away) + " " + away,
                    line_color='#EF4444'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='#1A3A5C',
                        radialaxis=dict(visible=True, gridcolor='rgba(45,212,191,0.2)'),
                        angularaxis=dict(gridcolor='rgba(45,212,191,0.2)'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=380,
                    legend=dict(bgcolor='#1A3A5C')
                )
                st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — SIMULATION
# ══════════════════════════════════════════════════════
elif page == "🏆 Simulation":

    st.markdown("# 🏆 Simulation CDM 2026")
    st.markdown("Modèle : **" + model_label + "** · Format officiel — 48 équipes · 12 groupes · 32 qualifiés")
    st.divider()

    if st.button("🚀 Lancer la simulation complète", type="primary", use_container_width=True):
        try:
            with st.spinner("⏳ Simulation en cours — environ 30 secondes..."):
                rounds, vainqueur, resultats_groupes, meilleurs_tiers = simuler_tournoi(
                    model, team_stats, model_name, features_df)

            st.balloons()

            st.markdown(
                "<div class='card-winner'>"
                "<p style='color:#0D1B2A; margin:0; font-size:1rem; font-weight:bold;'>👑 VAINQUEUR CDM 2026</p>"
                "<h1 style='color:#0D1B2A; margin:8px 0; font-size:3rem;'>"
                + flag(vainqueur) + " " + vainqueur
                + "</h1>"
                "<p style='color:#0D1B2A; margin:0;'>Prédit par " + model_label + "</p>"
                "</div>",
                unsafe_allow_html=True
            )

            st.divider()

            st.markdown("### 📋 Phase de groupes — Qualifiés")
            cols = st.columns(4)
            for i, (groupe, qualifies) in enumerate(resultats_groupes.items()):
                with cols[i % 4]:
                    q1 = flag(qualifies[0]) + " " + qualifies[0]
                    q2 = flag(qualifies[1]) + " " + qualifies[1]
                    st.markdown(
                        "<div class='card'>"
                        "<p style='color:#2DD4BF; font-weight:bold; margin:0;'>Groupe " + groupe + "</p>"
                        "<p style='color:white; margin:4px 0;'>🥇 " + q1 + "</p>"
                        "<p style='color:#94A3B8; margin:0;'>🥈 " + q2 + "</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )

            tiers_str = " · ".join([flag(t) + " " + t for t in meilleurs_tiers])
            st.markdown("**🔄 8 meilleurs 3èmes repêchés :** " + tiers_str)
            st.divider()

            round_icons = {
                'Seizièmes'   : '⚔️',
                'Huitièmes'   : '🔥',
                'Quarts'      : '💥',
                'Demi-finales': '⭐',
                'Finale'      : '🏆'
            }

            for round_name, matchs in rounds.items():
                if matchs:
                    icon = round_icons.get(round_name, '⚽')
                    st.markdown("### " + icon + " " + round_name)
                    nb_cols = min(len(matchs), 4)
                    cols = st.columns(nb_cols)
                    for idx, match_data in enumerate(matchs):
                        h, a, w = match_data
                        loser = a if w == h else h
                        with cols[idx % nb_cols]:
                            st.markdown(
                                "<div class='card' style='border-left:3px solid #10B981;'>"
                                "<p style='margin:0; font-size:0.9rem; color:#2DD4BF; font-weight:bold;'>"
                                + flag(w) + " " + w + " ✅</p>"
                                "<p style='margin:0; font-size:0.8rem; color:#64748B;'>"
                                "vs " + flag(loser) + " " + loser + "</p>"
                                "</div>",
                                unsafe_allow_html=True
                            )
                    st.markdown("")

        except Exception as e:
            st.error("Erreur : " + str(e))
