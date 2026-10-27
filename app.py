import streamlit as st
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="MediNet AI Dashboard", page_icon="🩺", layout="wide")

# --- 2. CUSTOM CSS FOR A PREMIUM LOOK ---
st.markdown("""
<style>
    /* Main container styling */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    
    /* Header styling */
    .main-title { text-align: center; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { text-align: center; color: #6B7280; margin-bottom: 2rem; }
    
    /* Card styling */
    div.css-1r6slb0, div.css-12oz5g7 {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] { font-size: 2.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('bayes_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
infer = VariableElimination(model)

# --- 4. HEADER ---
st.markdown("<h1 class='main-title'>🩺 MediNet AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Bidirectional Inference on a Bayesian Network</p>", unsafe_allow_html=True)

# ==========================================
# ROW 1: CAUSAL (LEFT) & EVIDENTIAL (RIGHT)
# ==========================================
col1, col_gap, col2 = st.columns([1, 0.1, 1])

# --- LEFT COLUMN: CAUSAL (Forward) ---
with col1:
    st.markdown("### 🎯 Causal Inference")
    st.caption("Predict disease risk from known risk factors.")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            age = st.selectbox("Age Group", ['Young', 'Middle', 'Old'])
            smoke = st.selectbox("Smoking Status", ['No', 'Yes'])
        with c2:
            exer = st.selectbox("Exercise Routine", ['Regular', 'Sedentary'])
            fam = st.selectbox("Family History", ['No', 'Yes'])
        
        # Inference
        ev_causal = {'Age': age, 'Smoking': smoke, 'Exercise': exer, 'FamilyHistory': fam}
        res_causal = infer.query(['HeartDisease'], evidence=ev_causal)
        p_hd = float(res_causal.values[1])
        
        st.divider()
        st.metric(label="Risk of Heart Disease", value=f"{p_hd*100:.1f}%")
        st.progress(p_hd)
        
        if p_hd < 0.15:
            st.success("🟢 **Low Risk** - Maintain healthy lifestyle.")
        elif p_hd < 0.45:
            st.warning("🟡 **Moderate Risk** - Recommend routine screening.")
        else:
            st.error("🔴 **High Risk** - Immediate cardiology consult advised.")

# --- RIGHT COLUMN: EVIDENTIAL (Backward) ---
with col2:
    st.markdown("### 🔍 Evidential Inference")
    st.caption("Diagnose hidden diseases from observed symptoms.")
    
    with st.container(border=True):
        c3, c4, c5 = st.columns(3)
        with c3: cp = st.selectbox("Chest Pain", ['No', 'Yes'])
        with c4: sob = st.selectbox("Shortness of Breath", ['No', 'Yes'])
        with c5: fat = st.selectbox("Fatigue", ['No', 'Yes'])
        
        # Inference
        ev_evid = {'ChestPain': cp, 'ShortnessBreath': sob, 'Fatigue': fat}
        res_evid_hd = infer.query(['HeartDisease'], evidence=ev_evid)
        res_evid_htn = infer.query(['Hypertension'], evidence=ev_evid)
        
        p_hd_evid = float(res_evid_hd.values[1])
        p_htn_evid = float(res_evid_htn.values[1])
        
        st.divider()
        st.metric(label="P(Heart Disease | Symptoms)", value=f"{p_hd_evid*100:.1f}%")
        st.progress(p_hd_evid)
        st.metric(label="P(Hypertension | Symptoms)", value=f"{p_htn_evid*100:.1f}%")
        st.progress(p_htn_evid)

st.divider()

# ==========================================
# ROW 2: NETWORK GRAPH & D-SEPARATION
# ==========================================
col_graph, col_sep = st.columns([1.5, 1])

# --- GRAPH VISUALIZATION ---
with col_graph:
    st.markdown("### 🕸️ Network Structure (DAG)")
    
    G = nx.DiGraph(model.edges())
    
    # MANUALLY DEFINING PERFECT NODE POSITIONS
    pos = {
        'Age': (-3, 2), 'Smoking': (-1.5, 2), 'Exercise': (0, 2), 'FamilyHistory': (2, 2),
        'Hypertension': (-1.5, 0.5),
        'HeartDisease': (0.5, -1),
        'ChestPain': (-1.5, -2.5), 'ShortnessBreath': (0.5, -2.5), 'Fatigue': (2.5, -2.5)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw edges first so they sit behind the nodes
    nx.draw_networkx_edges(G, pos, edge_color='#9CA3AF', arrows=True, arrowsize=25, width=2, ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#1E3A8A', node_size=4500, ax=ax, alpha=0.9)
    
    # Draw labels (white text inside dark blue circles)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold', ax=ax)
    
    plt.axis('off')
    st.pyplot(fig)

# --- D-SEPARATION ---
with col_sep:
    st.markdown("### 🧬 d-Separation Proof")
    st.caption("Verifying conditional independence via a collider.")
    
    with st.container(border=True):
        st.info("**Claim:** Chest Pain is independent of Smoking **given** Heart Disease.\n\n"
                "Because *Heart Disease* is a **Collider**, observing it blocks the path.")
        
        if st.button("Run Verification", use_container_width=True, type="primary"):
            q1 = infer.query(['ChestPain'], evidence={'HeartDisease':'Yes', 'Smoking':'Yes'})
            q2 = infer.query(['ChestPain'], evidence={'HeartDisease':'Yes', 'Smoking':'No'})
            
            p1 = float(q1.values[1])
            p2 = float(q2.values[1])
            
            st.markdown("**Scenario A:** Patient smokes & has Heart Disease")
            st.code(f"P(Chest Pain) = {p1:.4f}")
            
            st.markdown("**Scenario B:** Patient does NOT smoke & has Heart Disease")
            st.code(f"P(Chest Pain) = {p2:.4f}")
            
            diff = abs(p1 - p2)
            if diff < 1e-3:
                st.success(f"✓ **d-separation Confirmed** (Difference: {diff:.6f})\n\nSmoking gives no new information about Chest Pain.")
            else:
                st.error("✗ Values differ. Network failed.")