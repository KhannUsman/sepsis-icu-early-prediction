"""Sepsis Early Warning - Streamlit app.  Run:  streamlit run app/streamlit_app.py

The prediction runs ONLY when the "Predict" button is pressed.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.predict import load_bundle, predict_patient

st.set_page_config(page_title="Sepsis Early Warning", page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ glass style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
header[data-testid="stHeader"], footer, [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {display:none !important;}
.stApp {
  background:
    radial-gradient(circle at 8% 8%,  rgba(255,138,128,.55) 0, transparent 38%),
    radial-gradient(circle at 92% 12%, rgba(129,170,255,.50) 0, transparent 40%),
    radial-gradient(circle at 60% 95%, rgba(255,214,153,.55) 0, transparent 42%),
    linear-gradient(160deg,#fff5f4 0%,#f1f5ff 100%);
  background-attachment: fixed;
}
.block-container {max-width: 1250px; padding-top: 1.6rem; padding-bottom: 2rem;}

/* glass cards */
.st-key-hero, .st-key-inputs, .st-key-result, .st-key-perf1, .st-key-perf2, .st-key-extra {
  background: rgba(255,255,255,.42);
  backdrop-filter: blur(22px) saturate(170%);
  -webkit-backdrop-filter: blur(22px) saturate(170%);
  border: 1px solid rgba(255,140,130,.38);
  border-radius: 26px;
  box-shadow: 0 10px 40px rgba(60,40,80,.12), 0 0 22px rgba(229,57,53,.10),
              inset 0 1px 0 rgba(255,255,255,.9);
  padding: 22px 26px;
  transition: box-shadow .25s ease, border-color .25s ease;
}
.st-key-hero:hover, .st-key-inputs:hover, .st-key-result:hover, .st-key-perf1:hover,
.st-key-perf2:hover, .st-key-extra:hover {
  border-color: rgba(255,120,110,.6);
  box-shadow: 0 12px 44px rgba(60,40,80,.14), 0 0 30px rgba(229,57,53,.18),
              inset 0 1px 0 rgba(255,255,255,.9);
}
.st-key-hero {padding: 12px 30px;}
.hero-title {font-size: 2.35rem; font-weight: 800; line-height: 1.1; margin: 0;
  background: linear-gradient(90deg,#b71c1c,#e53935 55%,#ff7043); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}
.hero-sub {color:#4a5568; font-size: 1.02rem; margin: 8px 0 12px 0; max-width: 640px;}
.chip {display:inline-block; padding: 5px 14px; margin: 0 8px 6px 0; border-radius: 999px; font-size:.82rem;
  font-weight:600; color:#7f1d1d; background: rgba(255,255,255,.6); border:1px solid rgba(255,255,255,.9);
  backdrop-filter: blur(8px);}
.sect {font-size:.78rem; font-weight:700; letter-spacing:1.2px; color:#b71c1c; margin: 6px 0 4px 2px;}

/* inputs */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {
  background: rgba(255,255,255,.65) !important; border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,.9) !important;}
[data-testid="stNumberInput"] label p {font-size:.82rem; font-weight:600; color:#2d3748;}

/* sliders under the inputs */
[data-testid="stSlider"] {margin-top: -14px; padding: 0 6px;}
[data-testid="stSliderThumbValue"] {display:none !important;}
[data-testid="stSliderTickBar"] {display:none !important;}
[data-testid="stSlider"] [role="group"] > div > div:nth-child(2) {background:#fff !important;
  border:3px solid #e53935 !important; box-shadow: 0 2px 8px rgba(183,28,28,.4) !important;}
.rb-txt {display:flex; justify-content:space-between; font-size:.68rem; color:#6b7280; margin: -16px 6px 12px 6px;}
.rb-txt .rb-state {font-weight:700;}
.rb-ok {color:#17653a;} .rb-warn {color:#b45309;} .rb-bad {color:#b91c1c;}
.legend {font-size:.74rem; color:#4a5568; margin: 2px 0 6px 2px;}
.legend i {display:inline-block; width:10px; height:10px; border-radius:3px; margin: 0 4px 0 10px; vertical-align:-1px;}

/* buttons */
div.stButton > button {border-radius: 14px; font-weight: 700; height: 2.9rem;
  background: rgba(255,255,255,.65); border: 1px solid rgba(255,255,255,.95); color:#7f1d1d;
  backdrop-filter: blur(8px); transition: all .15s;}
div.stButton > button:hover {transform: translateY(-1px); border-color:#e53935; color:#b71c1c;}
div.stButton > button[kind="primary"] {background: linear-gradient(135deg,#e53935,#b71c1c); color:#fff; border:none;
  box-shadow: 0 8px 22px rgba(183,28,28,.35); font-size: 1.05rem;}
div.stButton > button[kind="primary"]:hover {transform: translateY(-2px); color:#fff;
  box-shadow: 0 12px 28px rgba(183,28,28,.45);}

/* result */
.badge {display:inline-block; padding: 8px 26px; border-radius: 999px; color:#fff; font-weight:800; font-size:1.1rem;
  letter-spacing:.4px; box-shadow: 0 6px 18px rgba(0,0,0,.18);}
.b-Low {background: linear-gradient(135deg,#2ecc71,#1e9e55);}
.b-Elevated {background: linear-gradient(135deg,#ffb300,#e68a00);}
.b-High {background: linear-gradient(135deg,#ef5350,#b71c1c);}
.msg {border-radius: 16px; padding: 12px 16px; margin: 14px 0 8px 0; font-size:.95rem; color:#2d3748;
  backdrop-filter: blur(10px);}
.m-Low {background: rgba(46,204,113,.18); border:1px solid rgba(46,204,113,.45);}
.m-Elevated {background: rgba(255,179,0,.20); border:1px solid rgba(255,179,0,.5);}
.m-High {background: rgba(239,83,80,.18); border:1px solid rgba(239,83,80,.5);}
.stale {border-radius: 12px; padding: 8px 14px; margin: 0 0 10px 0; font-size:.85rem; font-weight:600;
  background: rgba(255,179,0,.22); border:1px solid rgba(255,179,0,.55); color:#7a4b00;}
.sign {display:inline-block; padding: 6px 12px; margin: 0 6px 8px 0; border-radius: 12px; font-size:.82rem;
  font-weight:600; backdrop-filter: blur(8px);}
.s-ok {background: rgba(46,204,113,.18); color:#17653a; border:1px solid rgba(46,204,113,.4);}
.s-bad {background: rgba(239,83,80,.18); color:#a31515; border:1px solid rgba(239,83,80,.5);}
.empty {text-align:center; padding: 70px 10px; color:#6b7280;}
.empty .big {font-size: 3.2rem;}
.foot {text-align:center; color:#6b7280; font-size:.8rem; margin-top: 18px;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {border-radius: 999px; padding: 6px 18px; background: rgba(255,255,255,.45);}
.stTabs [aria-selected="true"] {background: rgba(255,255,255,.9); color:#b71c1c;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()
m = bundle["metrics"]

# ------------------------------------------------------------------ field definitions
# min/max  = what the box accepts.
# ok       = normal range (green)       severe = outside this is critical (red); between = watch (amber)
def F(key, label, short, lo, hi, ok, severe, step, fmt, typ):
    return dict(key=key, label=label, short=short, min=lo, max=hi, ok=ok, severe=severe,
                step=step, fmt=fmt, typ=typ)


FIELDS = {f["key"]: f for f in [
    F("HR", "Heart rate (bpm)", "HR", 20, 250, (50, 90), (40, 130), 1, "%d", int),
    F("Resp", "Resp. rate (/min)", "Resp", 4, 60, (12, 21), (8, 30), 1, "%d", int),
    F("Temp", "Temperature (°C)", "Temp", 30.0, 43.0, (36.0, 38.0), (35.0, 39.5), 0.1, "%.1f", float),
    F("SBP", "Systolic BP (mmHg)", "SBP", 40, 300, (101, 160), (90, 190), 1, "%d", int),
    F("MAP", "Mean arterial BP", "MAP", 20, 200, (65, 110), (55, 130), 1, "%d", int),
    F("O2Sat", "O2 saturation (%)", "O2", 40, 100, (92, 100), (88, 100), 1, "%d", int),
    F("Lactate", "Lactate", "Lactate", 0.3, 25.0, (0.3, 2.0), (0.3, 4.0), 0.1, "%.1f", float),
    F("WBC", "WBC", "WBC", 0.5, 100.0, (4.0, 12.0), (2.0, 20.0), 0.5, "%.1f", float),
    F("Creatinine", "Creatinine", "Creat.", 0.2, 20.0, (0.2, 1.2), (0.2, 2.5), 0.1, "%.2f", float),
    F("Platelets", "Platelets", "Plt", 5, 1000, (150, 450), (100, 600), 5, "%d", int),
    F("Age", "Age (years)", "Age", 18, 90, (18, 90), (18, 90), 1, "%d", int),
    F("ICULOS", "Hours in ICU", "ICU h", 1, 336, (1, 336), (1, 336), 1, "%d", int),
]}
SIGN_KEYS = [k for k in FIELDS if k not in ("Age", "ICULOS")]     # the 10 warning-sign checks

NORMAL = dict(HR=80, Resp=16, Temp=37.0, SBP=120, MAP=85, O2Sat=97,
              Lactate=1.0, WBC=8.0, Creatinine=1.0, Platelets=250, Age=60, ICULOS=10)
SEPTIC = dict(HR=118, Resp=24, Temp=38.4, SBP=96, MAP=64, O2Sat=95,
              Lactate=2.8, WBC=14.0, Creatinine=1.6, Platelets=140, Age=70, ICULOS=92)

# ------------------------------------------------------------------ session state
for k, v in NORMAL.items():
    st.session_state.setdefault(f"in_{k}", v)
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])

# Safety net: a value outside the allowed range can never stay in the boxes.
for k, f in FIELDS.items():
    raw = st.session_state[f"in_{k}"]
    try:
        raw = f["typ"](raw)
    except (TypeError, ValueError):
        raw = NORMAL[k]
    st.session_state[f"in_{k}"] = f["typ"](min(max(raw, f["min"]), f["max"]))
    st.session_state[f"sl_{k}"] = st.session_state[f"in_{k}"]      # slider always mirrors the box


def load_example(values):
    for k, v in values.items():
        st.session_state[f"in_{k}"] = v
        st.session_state[f"sl_{k}"] = v
    st.session_state["result"] = None      # old result is cleared, user must press Predict


def status(key, val):
    f = FIELDS[key]
    if f["ok"][0] <= val <= f["ok"][1]:
        return "ok"
    if f["severe"][0] <= val <= f["severe"][1]:
        return "warn"
    return "bad"


def slider_css():
    """Paint every slider track with its own green / amber / red zones."""
    red, amb, grn = "#ef9a9a", "#ffe082", "#a5d6a7"
    out = []
    for k, f in FIELDS.items():
        lo, hi = f["min"], f["max"]
        pct = lambda x: max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))
        a, b, c, d = pct(f["severe"][0]), pct(f["ok"][0]), pct(f["ok"][1]), pct(f["severe"][1])
        if f["ok"] == (lo, hi):
            grad = f"linear-gradient(90deg,{grn},{grn})"
        else:
            grad = (f"linear-gradient(90deg,{red} 0 {a:.1f}%,{amb} {a:.1f}% {b:.1f}%,{grn} {b:.1f}% {c:.1f}%,"
                    f"{amb} {c:.1f}% {d:.1f}%,{red} {d:.1f}% 100%)")
        out.append(f".st-key-sl_{k} [role=\"group\"] > div > div:first-child {{background: {grad} !important;}}")
    st.markdown("<style>" + "\n".join(out) + "</style>", unsafe_allow_html=True)


def sync_from_slider(key):
    st.session_state[f"in_{key}"] = st.session_state[f"sl_{key}"]


def field(col, key):
    f = FIELDS[key]
    with col:
        st.number_input(f["label"], min_value=f["min"], max_value=f["max"], step=f["step"],
                        format=f["fmt"], key=f"in_{key}",
                        help=f"Allowed: {f['min']:g} to {f['max']:g}. Normal: {f['ok'][0]:g} to {f['ok'][1]:g}.")
        st.slider(f["label"], min_value=f["min"], max_value=f["max"], step=f["step"],
                  key=f"sl_{key}", label_visibility="collapsed", on_change=sync_from_slider, args=(key,))
        word, s_ = "", "ok"
        if f["ok"] != (f["min"], f["max"]):
            s_ = status(key, st.session_state[f"in_{key}"])
            word = {"ok": "normal", "warn": "watch", "bad": "critical"}[s_]
        st.markdown(f'<div class="rb-txt"><span>{f["min"]:g}</span>'
                    f'<span class="rb-state rb-{s_}">{word}</span><span>{f["max"]:g}</span></div>',
                    unsafe_allow_html=True)


def current_vals():
    return {k: st.session_state[f"in_{k}"] for k in NORMAL}


slider_css()

# ------------------------------------------------------------------ hero
with st.container(key="hero"):
    c1, c2 = st.columns([1, 3.3], vertical_alignment="center")
    c1.image(str(ROOT / "app" / "assets" / "sepsis.png"), width=230)
    c2.markdown(f"""
      <div class="hero-title">ICU Sepsis Early Warning</div>
      <div class="hero-sub">Enter the patient's key signs and press <b>Predict</b>. The model looks for
      sepsis up to 6 hours before the clinical diagnosis.</div>
      <span class="chip">⏱ 6-hour early warning</span>
      <span class="chip">🧠 {bundle['model_name'].replace('_', ' ').title()}</span>
      <span class="chip">📈 ROC-AUC {m['test_roc_auc']:.2f}</span>
      <span class="chip">🎯 Recall {m['test_recall']:.0%} (balanced) / {bundle.get('metrics_high_recall', m)['test_recall']:.0%} (high sensitivity)</span>""", unsafe_allow_html=True)

st.write("")
tab_pred, tab_model = st.tabs(["🩺  Predict", "📊  Model performance"])

# ------------------------------------------------------------------ predict tab
with tab_pred:
    e1, e2, e3 = st.columns([1, 1, 3], vertical_alignment="center")
    e1.button("Healthy example", on_click=load_example, args=(NORMAL,), width="stretch")
    e2.button("Sepsis-like example", on_click=load_example, args=(SEPTIC,), width="stretch")
    mode_label = e3.radio("Alert mode", ["Balanced", "High sensitivity"], horizontal=True, key="mode_label",
                          help="Balanced: best F2 threshold. High sensitivity: catches more sepsis cases "
                               "(recall about 70%) but gives many more false alarms.")
    MODE = "high_sensitivity" if mode_label == "High sensitivity" else "balanced"

    left, right = st.columns([1.2, 1], gap="large")

    with left, st.container(key="inputs"):
        st.markdown('<div class="legend"><b>Bar guide:</b><i style="background:#a5d6a7"></i>normal'
                    '<i style="background:#ffe082"></i>watch<i style="background:#ef9a9a"></i>critical'
                    '&nbsp;· ends = min / max you can enter &nbsp;· drag or click the bar</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sect">VITAL SIGNS</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, k in zip(cols, ["HR", "Resp", "Temp"]):
            field(col, k)
        cols = st.columns(3)
        for col, k in zip(cols, ["SBP", "MAP", "O2Sat"]):
            field(col, k)

        st.markdown('<div class="sect">LAB RESULTS</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for col, k in zip(cols, ["Lactate", "WBC", "Creatinine", "Platelets"]):
            field(col, k)

        st.markdown('<div class="sect">PATIENT</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for col, k in zip(cols, ["Age", "ICULOS"]):
            field(col, k)

        submitted = st.button("Predict sepsis risk", type="primary", width="stretch")

    if submitted:                                   # the ONLY place where we predict
        vals = current_vals()
        with st.spinner("Predicting..."):
            res = predict_patient(vals, bundle, MODE)
            # what-if: put ONE value back to normal and see how much the score drops
            base, impact = res["probability"], []
            for k in SIGN_KEYS:
                if status(k, vals[k]) != "ok":
                    alt = dict(vals); alt[k] = NORMAL[k]
                    drop = (base - predict_patient(alt, bundle, MODE)["probability"]) * 100
                    impact.append((FIELDS[k]["short"], drop))
        st.session_state["result"] = {"res": res, "vals": vals, "impact": impact, "mode": mode_label}
        st.session_state["history"].append({
            "Time": datetime.now().strftime("%H:%M:%S"), **vals,
            "Risk %": round(res["probability"] * 100, 1), "Level": res["level"], "Mode": mode_label})

    # ---------- result
    with right, st.container(key="result"):
        out = st.session_state["result"]
        if out is None:
            st.markdown("""<div class="empty"><div class="big">🫀</div>
              <h4 style="margin:6px 0">No prediction yet</h4>
              Fill the values and press <b>Predict sepsis risk</b>.</div>""", unsafe_allow_html=True)
        else:
            res, v = out["res"], out["vals"]
            if out.get("mode") != mode_label:
                st.markdown('<div class="stale">⚠ Alert mode changed after this result. '
                            'Press <b>Predict sepsis risk</b> again to update.</div>', unsafe_allow_html=True)
            if v != current_vals():
                st.markdown('<div class="stale">⚠ Values changed after this result. '
                            'Press <b>Predict sepsis risk</b> again to update.</div>', unsafe_allow_html=True)
            score, thr = res["probability"] * 100, res["threshold"] * 100
            color = {"Low": "#2ecc71", "Elevated": "#ffb300", "High": "#e53935"}[res["level"]]
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score, number={"valueformat": ".0f", "font": {"size": 56}},
                title={"text": "Sepsis risk score"},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                       "bar": {"color": color, "thickness": .32},
                       "bgcolor": "rgba(255,255,255,.35)", "borderwidth": 0,
                       "steps": [{"range": [0, .8 * thr], "color": "rgba(46,204,113,.28)"},
                                 {"range": [.8 * thr, thr], "color": "rgba(255,179,0,.30)"},
                                 {"range": [thr, 100], "color": "rgba(239,83,80,.30)"}],
                       "threshold": {"line": {"color": "#1f2937", "width": 4}, "thickness": .85, "value": thr}}))
            fig.update_layout(height=270, margin=dict(l=25, r=25, t=50, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"color": "#1f2937", "family": "Inter"})
            st.plotly_chart(fig, width="stretch", theme=None)
            st.markdown(f'<div style="text-align:center"><span class="badge b-{res["level"]}">'
                        f'{res["level"]} risk</span></div>', unsafe_allow_html=True)
            text = {"High": "<b>Sepsis alert.</b> Consider blood cultures, lactate check, antibiotics and fluids "
                            "as per hospital protocol. Doctor review needed.",
                    "Elevated": "<b>Watch closely.</b> Risk is near the alert line. Repeat vitals and labs soon.",
                    "Low": "<b>Low risk</b> right now. Keep routine monitoring."}[res["level"]]
            st.markdown(f'<div class="msg m-{res["level"]}">{text}</div>', unsafe_allow_html=True)

            chips, bad = "", 0
            for k in SIGN_KEYS:
                flag = status(k, v[k]) != "ok"; bad += flag
                chips += f'<span class="sign {"s-bad" if flag else "s-ok"}">{FIELDS[k]["short"]} {v[k]:g}</span>'
            st.markdown(f'<div class="sect">WARNING SIGNS: {bad} OF {len(SIGN_KEYS)} ABNORMAL</div>{chips}',
                        unsafe_allow_html=True)

    # ---------- extra: what drives the score + history
    if out is not None:
        with st.container(key="extra"):
            x1, x2 = st.columns([1, 1], gap="large")
            with x1:
                st.markdown("**What is pushing the score up?**")
                st.caption("Each abnormal value is set back to normal, one at a time. "
                           "The bar shows how many score points would drop.")
                imp = sorted([i for i in out["impact"]], key=lambda t: t[1])
                if not imp:
                    st.info("All 10 warning signs are normal, so there is nothing to explain.")
                else:
                    fig2 = go.Figure(go.Bar(
                        x=[d for _, d in imp], y=[n for n, _ in imp], orientation="h",
                        marker_color=["#e53935" if d > 0.05 else "#9ca3af" for _, d in imp],
                        text=[f"{d:+.1f}" for _, d in imp], textposition="outside"))
                    fig2.update_layout(height=max(200, 38 * len(imp) + 90), margin=dict(l=10, r=50, t=10, b=45),
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       xaxis=dict(title="Score points", range=[min(0, min(d for _, d in imp) * 1.3),
                                                           max(1, max(d for _, d in imp) * 1.3)]),
                                       yaxis=dict(automargin=True), font={"family": "Inter"})
                    st.plotly_chart(fig2, width="stretch", theme=None)
                    if all(abs(d) < 0.05 for _, d in imp):
                        st.caption("The model does not react to these values on their own. It only sees 12 inputs "
                                   "and combines them, so it can stay flat until several values change together.")
            with x2:
                st.markdown("**This session's predictions**")
                if st.session_state["history"]:
                    hist = pd.DataFrame(st.session_state["history"])
                    st.dataframe(hist[["Time", "HR", "MAP", "Lactate", "ICULOS", "Risk %", "Level"]].iloc[::-1],
                                 hide_index=True, width="stretch", height=200)
                    d1, d2 = st.columns(2)
                    d1.download_button("Download history (CSV)", hist.to_csv(index=False).encode(),
                                       "sepsis_predictions.csv", "text/csv", width="stretch")
                    if d2.button("Clear history", width="stretch"):
                        st.session_state["history"] = []
                        st.rerun()
                else:
                    st.info("History is empty. Press Predict to add a row.")
                with st.expander("Exact values sent to the model"):
                    st.json(out["vals"])

# ------------------------------------------------------------------ model tab
with tab_model:
    p1, p2 = st.columns([1.1, 1], gap="large")
    with p1, st.container(key="perf1"):
        table = pd.read_csv(C.RESULTS_DIR / "comparison.csv")
        show = table[["model", "test_roc_auc", "test_pr_auc", "test_recall", "test_precision",
                      "test_f2"]].round(3)
        show.columns = ["Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F2"]
        st.markdown("**All models on the test set**")
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption(f"Selected: {bundle['model_name'].replace('_', ' ').title()} "
                   "(best mean of validation ROC-AUC, PR-AUC and F2 among all compared models).")
        if "hr_recall" in table.columns:
            hs = table.dropna(subset=["hr_recall"])[["model", "hr_recall", "hr_precision", "hr_specificity"]].round(3)
            hs.columns = ["Model", "Recall", "Precision", "Specificity"]
            st.markdown("**High-sensitivity mode (threshold for about 70% recall)**")
            st.dataframe(hs, hide_index=True, width="stretch")
            st.caption("More sepsis cases are found, but most alarms are false. Precision shows how many alarms are real.")
        img = C.FIG_DIR / "model_comparison_bars.png"
        if img.exists():
            st.image(str(img), width="stretch")
    with p2, st.container(key="perf2"):
        st.markdown("**Saved settings of the selected model**")
        if C.BEST_CONFIG_PATH.exists():
            cfg = json.load(open(C.BEST_CONFIG_PATH))
            st.json({"model": cfg["model_name"], "tuned_params": cfg["tuned_params"],
                     "threshold_balanced": round(cfg.get("final_threshold", cfg["threshold"]), 3),
                     "threshold_high_sensitivity": round(cfg.get("final_threshold_high_recall", 0), 3), "features": cfg["features"]}, expanded=1)
        img = C.FIG_DIR / f"{bundle['model_name']}_importance.png"
        if img.exists():
            st.image(str(img), width="stretch")

st.markdown('<div class="foot">Student project, not a medical device. '
            'Do not use for real patient care.</div>', unsafe_allow_html=True)