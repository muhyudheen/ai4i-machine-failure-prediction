"""Streamlit dashboard: AI4I 2020 machine-failure prediction.

Run locally:  streamlit run app.py
Everything shown is read from artefacts produced by the notebooks
(models/, reports/), or computed live with the same src/ code they use.
"""
import json

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import FAILURE_MODE_NAMES, FAILURE_MODES, FIG_DIR, MODELS_DIR, PRETTY, REPORTS_DIR, TYPE_ORDER
from src.cv import METRIC_NAMES, METRICS
from src.data import get_X_y
from src.explain import explain, make_explainer
from src.features import add_physics_features
from src.optimize import (RISK_BUDGET, RPM_GRID, TORQUE_GRID, conditions, power_w, risk_grid,
                          support_mask)
from src.pipeline import MODEL_NAMES, STRATEGY_NAMES
from src import plots

st.set_page_config(page_title="AI4I 2020 · Machine-failure risk", page_icon="⚙️", layout="wide")
plots.set_style()

SEQ_RANGE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


# ----------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    modes = joblib.load(MODELS_DIR / "mode_models.joblib")
    return model, modes, make_explainer(model)


@st.cache_data
def load_tables():
    meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    return (meta, pd.read_csv(REPORTS_DIR / "cv_summary.csv"), pd.read_csv(REPORTS_DIR / "oof_predictions.csv"),
            pd.read_csv(REPORTS_DIR / "operating_ranges.csv"), pd.read_csv(REPORTS_DIR / "mode_model_cv.csv"))


@st.cache_data
def load_xy():
    return get_X_y()


@st.cache_data
def cached_support():
    X, _ = load_xy()
    return support_mask(X)


model, mode_models, explainer = load_models()
meta, cv_summary, oof, ranges, mode_cv = load_tables()
X, y = load_xy()
support = cached_support()
THRESHOLD = meta["threshold"]
base = conditions(X)

# ----------------------------------------------------------------- header
st.title("Machine-failure risk from process parameters")
st.caption("AI4I 2020 Predictive Maintenance dataset (UCI 601): 10,000 runs, 3.39% failures. "
           f"Model: {meta['selected']}. All metrics are stratified 5-fold cross-validation, mean ± std.")
nested = meta["metrics_at_tuned_threshold_nested"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC (primary)", f"{meta['cv_headline_pr_auc'][0]:.3f}", f"± {meta['cv_headline_pr_auc'][1]:.3f}", delta_color="off", delta_arrow="off")
c2.metric("Recall at alarm threshold", f"{nested['recall'][0]:.1%}", f"± {nested['recall'][1]:.1%}", delta_color="off", delta_arrow="off")
c3.metric("Precision at alarm threshold", f"{nested['precision'][0]:.1%}", f"± {nested['precision'][1]:.1%}", delta_color="off", delta_arrow="off")
c4.metric("Alarm threshold on P(failure)", f"{THRESHOLD:.3f}", "tuned for recall ≥ 90%", delta_color="off", delta_arrow="off")

tab_perf, tab_xai, tab_sim, tab_ops = st.tabs(
    ["📊 Model performance", "🔍 Explainability", "🎛️ What-if simulator", "🧭 Operating ranges"])

# ================================================================= 1. performance
with tab_perf:
    st.subheader("Cross-validation results: model × imbalance strategy")
    eng = st.toggle("With physics features (power, temperature difference, overstrain)", value=True)
    t = cv_summary[cv_summary.engineered == eng].copy()
    t.insert(0, "Model", t.model.map(MODEL_NAMES))
    t.insert(1, "Imbalance strategy", t.strategy.map(STRATEGY_NAMES))
    for m in METRICS:
        t[METRIC_NAMES[m]] = t[f"{m}_mean"].map("{:.3f}".format) + " ± " + t[f"{m}_std"].map("{:.3f}".format)
    t = t.assign(_m=t.model.map({m: i for i, m in enumerate(MODEL_NAMES)}),
                 _s=t.strategy.map({s_: i for i, s_ in enumerate(STRATEGY_NAMES)})).sort_values(["_m", "_s"])
    st.dataframe(t[["Model", "Imbalance strategy"] + [METRIC_NAMES[m] for m in METRICS]],
                 hide_index=True, width="stretch")
    st.caption("PR-AUC is the selection metric (chance level = 0.034). F1/recall/precision/accuracy use threshold 0.5. "
               "Accuracy is shown only to make the point: always predicting 'no failure' scores 96.6%. "
               "The selected model is a nested-CV-tuned Random Forest (PR-AUC 0.915 ± 0.006); see reports/metrics.md.")

    st.subheader("Threshold explorer (selected model, out-of-fold predictions)")
    thr = st.slider("Decision threshold on P(failure)", 0.01, 0.95, float(round(THRESHOLD, 3)), 0.001, format="%.3f",
                    help="Default = the deployed threshold (highest threshold with recall ≥ 90%).")
    yt, pf = oof["y"].values, oof["final"].values
    pred = pf >= thr
    tp, fp = int((pred & (yt == 1)).sum()), int((pred & (yt == 0)).sum())
    fn, tn = int((~pred & (yt == 1)).sum()), int((~pred & (yt == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / (tp + fn)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall (failures caught)", f"{rec:.1%}", f"{tp} of {tp + fn}", delta_color="off", delta_arrow="off")
    k2.metric("Precision (alarms that are real)", f"{prec:.1%}", f"{fp} false alarms", delta_color="off", delta_arrow="off")
    k3.metric("F1 (failure)", f"{2 * prec * rec / max(prec + rec, 1e-9):.3f}")
    k4.metric("Missed failures", f"{fn}", f"of {len(yt):,} runs", delta_color="off", delta_arrow="off")

    # One curve per model family, coloured as in the notebook figures (LR blue, RF orange, XGBoost green).
    # The untuned RF is left out: it overlaps the tuned one almost exactly and only adds clutter.
    xgb_col = next(c for c in oof.columns if c.startswith("xgb_"))
    lr_col = next(c for c in oof.columns if c.startswith("logreg_"))
    curves = {"Random Forest, tuned (selected)": "final",
              f"XGBoost ({STRATEGY_NAMES[xgb_col.split('_', 1)[1]].lower()})": xgb_col,
              f"Logistic Regression ({STRATEGY_NAMES[lr_col.split('_', 1)[1]].lower()})": lr_col}
    curve_colors = [plots.SERIES[1], plots.SERIES[2], plots.SERIES[0]]
    pr_rows, roc_rows = [], []
    for name, col in curves.items():
        p, r, th = precision_recall_curve(yt, oof[col])
        pr_rows.append(pd.DataFrame({"model": name, "recall": r[:-1], "precision": p[:-1], "threshold": th}))
        fpr, tpr, th2 = roc_curve(yt, oof[col])
        roc_rows.append(pd.DataFrame({"model": name, "fpr": fpr[1:], "tpr": tpr[1:], "threshold": th2[1:]}))
    pr_df, roc_df = pd.concat(pr_rows), pd.concat(roc_rows)
    color = alt.Color("model:N", title=None, scale=alt.Scale(domain=list(curves), range=curve_colors),
                      legend=alt.Legend(orient="bottom", columns=1, labelLimit=0, symbolStrokeWidth=3))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["recall"], empty=False)
    pr_line = alt.Chart(pr_df).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1.02])),
        color=color)
    pr_pts = alt.Chart(pr_df).mark_point(size=60, filled=True).encode(
        x="recall:Q", y="precision:Q", color=color,
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=["model", alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f"),
                 alt.Tooltip("threshold:Q", format=".3f")]).add_params(nearest)
    # Current operating point in ink (not a series colour), so it can't be mistaken for part of a curve
    here = alt.Chart(pd.DataFrame({"recall": [rec], "precision": [prec]})).mark_point(
        shape="diamond", size=220, filled=True, color=plots.INK, stroke=plots.SURFACE, strokeWidth=2, opacity=1).encode(
        x="recall:Q", y="precision:Q", tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")])
    roc = alt.Chart(roc_df).mark_line(strokeWidth=2).encode(
        x=alt.X("fpr:Q", title="False-positive rate"), y=alt.Y("tpr:Q", title="True-positive rate", scale=alt.Scale(domain=[0, 1.01])),
        color=color, tooltip=["model", alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f"),
                              alt.Tooltip("threshold:Q", format=".3f")])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=plots.MUTED).encode(x="x", y="y")

    cm = pd.DataFrame({"actual": ["No", "No", "Yes", "Yes"], "predicted": ["No", "Yes", "No", "Yes"],
                       "n": [tn, fp, fn, tp]})
    cm["share"] = cm.n / cm.groupby("actual").n.transform("sum")
    cm["label"] = cm.n.map("{:,}".format) + "\n" + cm.share.map("{:.1%}".format)
    order = ["No", "Yes"]
    cm_base = alt.Chart(cm).encode(
        x=alt.X("predicted:N", title="Predicted failure?", sort=order, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("actual:N", title="Actual failure?", sort=order, axis=alt.Axis(labelAngle=0)))
    cm_chart = (cm_base.mark_rect(stroke=plots.SURFACE, strokeWidth=2).encode(
                    color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=SEQ_RANGE), legend=None),
                    tooltip=["actual", "predicted", alt.Tooltip("n:Q", format=","), alt.Tooltip("share:Q", format=".1%")])
                + cm_base.mark_text(fontSize=14, lineBreak="\n").encode(
                    text="label:N",
                    color=alt.condition("datum.share > 0.55", alt.value("white"), alt.value(plots.INK))))

    g1, g2, g3 = st.columns([1.15, 1.15, 0.9])
    with g1:
        st.markdown("**Precision-recall** (◆ = current threshold)")
        st.altair_chart((pr_line + pr_pts + here).properties(height=340), width="stretch")
    with g2:
        st.markdown("**ROC**")
        st.altair_chart((roc + diag).properties(height=340), width="stretch")
    with g3:
        st.markdown("**Confusion matrix** at this threshold")
        st.altair_chart(cm_chart.properties(height=260), width="stretch")
        st.caption("Counts, and % of each actual row.")
    st.caption("Out-of-fold predictions: every run is scored by a model that never trained on it. The selected model's "
               "hyper-parameters were chosen on all data, so these curves are very slightly optimistic. The headline numbers "
               "above come from fully nested CV.")

# ================================================================= 2. explainability
with tab_xai:
    st.subheader("What drives failure risk?")
    # Full-width, one figure per row: these are dense multi-panel figures and become unreadable side by side
    st.image(str(FIG_DIR / "13_shap_summary.png"), width="stretch", caption="SHAP summary (all 10,000 runs). Each dot is a run; "
             "colour = feature value; x = how much it moved P(failure).")
    st.image(str(FIG_DIR / "12_feature_importance.png"), width="stretch", caption="Left: impurity importance. Right: "
             "cross-validated permutation importance (drop in PR-AUC when a raw input is shuffled).")
    st.image(str(FIG_DIR / "15_partial_dependence.png"), width="stretch", caption="Partial dependence per product Type. "
             "The rpm curve is partly an artefact: it pairs high speed with each run's own torque (see notebook 03, §4).")
    st.markdown(
        "- **Torque and speed** matter most as raw inputs (largest PR-AUC drop when shuffled). The model uses them "
        "mainly through **power** and **tool wear × torque (overstrain)**.\n"
        "- **Power** has a two-sided effect: both too low and too high are risky (the power-failure band). A linear "
        "model cannot represent this, which is why logistic regression lags.\n"
        "- **Low temperature difference** combined with **low speed** drives heat-dissipation failures; this is why "
        "air temperature ranks high among raw inputs.\n"
        "- **Tool wear** is harmless until about 200 min, then risk rises about 40× (the random tool-wear-failure window).\n"
        "- Per-run explanations for any setting are in the **What-if simulator** tab.")

# ================================================================= 3. what-if
with tab_sim:
    st.subheader("Set the process parameters")
    left, right = st.columns([1, 1.35])
    with left:
        typ = st.radio("Product quality Type", TYPE_ORDER, index=0, horizontal=True,
                       format_func=lambda t: {"L": "L (low)", "M": "M (medium)", "H": "H (high)"}[t])
        air = st.slider(PRETTY["air_temp_k"], 295.0, 305.0, float(base["air_temp_k"]), 0.1)
        proc = st.slider(PRETTY["process_temp_k"], 305.0, 314.0, float(base["process_temp_k"]), 0.1)
        rpm = st.slider(PRETTY["rot_speed_rpm"], 1150, 2900, int(base["rot_speed_rpm"]), 10)
        torque = st.slider(PRETTY["torque_nm"], 3.0, 77.0, float(base["torque_nm"]), 0.5)
        wear = st.slider(PRETTY["tool_wear_min"], 0, 253, int(base["tool_wear_min"]), 1)

    row = pd.DataFrame([{"type": typ, "air_temp_k": air, "process_temp_k": proc, "rot_speed_rpm": rpm,
                         "torque_nm": torque, "tool_wear_min": wear}])
    p_fail = float(model.predict_proba(row)[:, 1][0])
    p_modes = pd.Series({m: float(mode_models[m].predict_proba(row)[:, 1][0]) for m in FAILURE_MODES})
    derived = add_physics_features(row).iloc[0]
    ri = np.abs(RPM_GRID - rpm).argmin(); ti = np.abs(TORQUE_GRID - torque).argmin()
    supported = bool(support[ti, ri])

    with right:
        if p_fail >= THRESHOLD:
            st.error(f"🚨 **ALARM**: P(failure) = **{p_fail:.1%}**, at or above the alarm threshold ({THRESHOLD:.1%}).")
        elif p_fail > RISK_BUDGET:
            st.warning(f"⚠️ **Elevated**: P(failure) = **{p_fail:.1%}**, above the {RISK_BUDGET:.0%} design risk budget "
                       f"but below the alarm threshold.")
        else:
            st.success(f"✅ **Within budget**: P(failure) = **{p_fail:.2%}** (≤ {RISK_BUDGET:.0%} risk budget).")
        if not supported:
            st.info("ℹ️ This speed/torque combination is **outside the training data** (speed and torque are tightly "
                    "linked in this dataset). The model is extrapolating, so treat the number with caution.")
        top = p_modes.idxmax()
        if p_modes[top] < RISK_BUDGET:
            # argmax over near-zero probabilities is noise; don't present it as a finding
            st.markdown(f"**Failure mode:** none expected (every mode below {RISK_BUDGET:.0%}).")
        else:
            st.markdown(f"**Most likely failure mode:** {FAILURE_MODE_NAMES[top]} ({top}), P = {p_modes[top]:.1%}")
        mdf = pd.DataFrame({"mode": [f"{m} · {FAILURE_MODE_NAMES[m].replace(' failure', '').lower()}" for m in FAILURE_MODES],
                            "p": p_modes.values, "p_label": [f"{v:.1%}" for v in p_modes.values],
                            "cv": mode_cv.set_index("mode").loc[FAILURE_MODES, "PR-AUC"].values})
        x_max = max(0.05, mdf.p.max() * 1.25)
        mode_base = alt.Chart(mdf).encode(
            y=alt.Y("mode:N", title=None, sort=None, axis=alt.Axis(labelLimit=0)),
            tooltip=[alt.Tooltip("mode:N"), alt.Tooltip("p:Q", format=".2%", title="P(mode)"),
                     alt.Tooltip("cv:N", title="Mode model CV PR-AUC")])
        mode_bars = mode_base.mark_bar(cornerRadiusEnd=4, height=16, color=plots.C_FAIL).encode(
            x=alt.X("p:Q", title="P(mode)", scale=alt.Scale(domain=[0, x_max]), axis=alt.Axis(format="%")))
        mode_text = mode_base.mark_text(align="left", dx=5, color=plots.INK_2).encode(x="p:Q", text="p_label:N")
        st.altair_chart((mode_bars + mode_text).properties(height=190), width="stretch")
        d1, d2, d3 = st.columns(3)
        d1.metric("Power", f"{derived.power_w / 1000:.2f} kW", "safe band 3.5–9 kW", delta_color="off", delta_arrow="off",
                  help="Power failure (PWF) occurs when power is below 3.5 kW or above 9 kW.")
        d2.metric("Temp. difference", f"{derived.temp_diff_k:.1f} K", "risk if < 8.6 K", delta_color="off", delta_arrow="off",
                  help="Heat-dissipation failure (HDF) occurs when ΔT < 8.6 K AND speed < 1,380 rpm.")
        d3.metric("Wear × torque", f"{derived.overstrain_nm_min:,.0f}", f"limit {OSF_LIMIT[typ]:,} min·Nm",
                  delta_color="off", delta_arrow="off",
                  help="Overstrain failure (OSF) occurs above 11,000 (L) / 12,000 (M) / 13,000 (H) min·Nm.")
        st.caption("The 'rule' references are the dataset's documented failure mechanisms (verified in notebook 03). "
                   "The model never saw them. It learned them from process parameters alone.")

    st.markdown("**Why this prediction?** SHAP contributions, in probability units: base rate + bars = P(failure)")
    sv_row = explain(model, row, explainer)
    shap.plots.waterfall(sv_row[0], max_display=10, show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 4.2)
    st.pyplot(fig, width="content")
    plt.close(fig)

# ================================================================= 4. operating ranges
with tab_ops:
    st.subheader("Recommended operating ranges per Type")
    st.markdown(f"Settings where predicted risk stays within the **{RISK_BUDGET:.0%} risk budget** at every tool wear "
                "up to the limit, **restricted to settings the data actually covers**. Ambient temperatures are at their medians.")
    show = pd.DataFrame({
        "Type": ranges.type,
        "Change tool before": ranges.tool_wear_max_min.map("{:.0f} min".format),
        "Power band": ranges.power_min_w.map("{:,.0f}".format) + "–" + ranges.power_max_w.map("{:,.0f} W".format),
        "Speed": ranges.rpm_min.map("{:,.0f}".format) + "–" + ranges.rpm_max.map("{:,.0f} rpm".format),
        "Torque": ranges.torque_min_nm.map("{:.1f}".format) + "–" + ranges.torque_max_nm.map("{:.1f} Nm".format),
        "Min speed if ΔT ≈ 8 K": ranges.min_rpm_if_temp_diff_8K.map("{:,.0f} rpm".format),
        "Robust set-point": ranges.setpoint_rpm.map("{:,.0f} rpm".format) + " · " + ranges.setpoint_torque_nm.map("{:.1f} Nm".format),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Speed and torque are tied together (ρ = −0.92), so the safe region is a power band within the observed "
               "operating band, not a rectangle: stay inside the power band and below the torque limit. Checked against "
               "the dataset's physics in notebook 03 (power band, heat-dissipation speed, tool-wear window and L/M overstrain "
               "limits recovered; Type H's torque limit is conservative because high-strain H runs are rare).")

    st.subheader("Explore the risk map")
    o1, o2, o3, o4 = st.columns(4)
    m_typ = o1.selectbox("Type", TYPE_ORDER, key="ops_type")
    m_wear = o2.slider(PRETTY["tool_wear_min"], 0, 253, 100, 5, key="ops_wear")
    m_air = o3.slider(PRETTY["air_temp_k"], 295.0, 305.0, float(base["air_temp_k"]), 0.1, key="ops_air")
    m_td = o4.slider(PRETTY["temp_diff_k"], 7.5, 12.0, float(round(base["process_temp_k"] - base["air_temp_k"], 1)), 0.1, key="ops_td")

    @st.cache_data
    def map_frame(typ_, wear_, air_, td_):
        P = risk_grid(model, typ_, base, tool_wear_min=wear_, air_temp_k=air_, process_temp_k=air_ + td_)
        R, T = np.meshgrid(RPM_GRID, TORQUE_GRID)
        dr, dt = np.diff(RPM_GRID)[0] / 2, np.diff(TORQUE_GRID)[0] / 2
        f = pd.DataFrame({"rpm": R.ravel(), "torque": T.ravel(), "p": P.ravel(), "sup": support.ravel()})
        f = f[f.sup].assign(rpm0=lambda d: d.rpm - dr, rpm1=lambda d: d.rpm + dr,
                            t0=lambda d: d.torque - dt, t1=lambda d: d.torque + dt,
                            power_kw=lambda d: power_w(d.rpm, d.torque) / 1000,
                            p_plot=lambda d: d.p.clip(1e-3, 1))
        return f

    f = map_frame(m_typ, m_wear, m_air, m_td)
    # Diverging scale centred on the risk budget: blue = within budget, grey ≈ budget, red = above it.
    risk_scale = alt.Scale(type="log", domain=[1e-3, 4e-3, RISK_BUDGET, 6e-2, 1],
                           range=["#1c5cab", "#86b6ef", "#c3c2b7", "#e66767", "#a32424"])
    x_rpm = alt.X("rpm0:Q", title=PRETTY["rot_speed_rpm"], scale=alt.Scale(domain=[1150, 2900], nice=False))
    y_tq = alt.Y("t0:Q", title=PRETTY["torque_nm"], scale=alt.Scale(domain=[3, 78], nice=False))
    heat = alt.Chart(f).mark_rect().encode(
        x=x_rpm, x2="rpm1:Q", y=y_tq, y2="t1:Q",
        color=alt.Color("p_plot:Q", title=["P(failure)", f"budget = {RISK_BUDGET:.0%}"], scale=risk_scale,
                        legend=alt.Legend(format=".1%", values=[1e-3, RISK_BUDGET, 0.1, 1], gradientLength=220)),
        tooltip=[alt.Tooltip("rpm:Q", format=".0f", title="Speed [rpm]"), alt.Tooltip("torque:Q", format=".1f", title="Torque [Nm]"),
                 alt.Tooltip("power_kw:Q", format=".2f", title="Power [kW]"), alt.Tooltip("p:Q", format=".2%", title="P(failure)")])
    rr = np.linspace(1150, 2900, 200)
    rules = pd.concat([pd.DataFrame({"rpm": rr, "torque": pw / (rr * np.pi / 30), "line": lab})
                       for pw, lab in ((3500, "PWF 3.5 kW"), (9000, "PWF 9 kW"))])
    rule_lines = alt.Chart(rules[rules.torque.between(3, 78)]).mark_line(strokeDash=[5, 4], color=plots.INK_2, strokeWidth=1).encode(
        x="rpm:Q", y="torque:Q", detail="line:N")
    sp = ranges.set_index("type").loc[m_typ]
    sp_df = pd.DataFrame({"rpm": [sp.setpoint_rpm], "torque": [sp.setpoint_torque_nm],
                          "label": [f"set-point {sp.setpoint_rpm:,.0f} rpm · {sp.setpoint_torque_nm:.1f} Nm"]})
    sp_mark = alt.Chart(sp_df).mark_point(shape="diamond", size=260, filled=True, color=plots.INK, stroke=plots.SURFACE,
                                          strokeWidth=2, opacity=1).encode(x="rpm:Q", y="torque:Q", tooltip=["label:N"])
    # Label drawn twice: a thick surface-coloured stroke first (halo), then the ink text, so it reads on any cell colour
    sp_halo = alt.Chart(sp_df).mark_text(align="left", dx=12, dy=-12, fontWeight="bold", stroke=plots.SURFACE,
                                         strokeWidth=4, strokeJoin="round").encode(x="rpm:Q", y="torque:Q", text="label:N")
    sp_text = alt.Chart(sp_df).mark_text(align="left", dx=12, dy=-12, color=plots.INK, fontWeight="bold").encode(
        x="rpm:Q", y="torque:Q", text="label:N")
    st.altair_chart((heat + rule_lines + sp_mark + sp_halo + sp_text).properties(height=460), width="stretch")
    st.caption(f"Only data-supported settings are drawn. Blue = within the {RISK_BUDGET:.0%} risk budget, red = above it. "
               f"◆ = the robust set-point for Type {m_typ} (from the table above). Dashed = the dataset's true power-failure "
               "limits (3.5 and 9 kW). Hover any cell for its numbers.")
