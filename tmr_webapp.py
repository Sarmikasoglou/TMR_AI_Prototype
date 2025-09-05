# tmr_webapp.py
# Streamlit TMR Formulator
# Run with: streamlit run tmr_webapp.py

import streamlit as st
from dataclasses import dataclass
import pulp
import pandas as pd

st.set_page_config(page_title="TMR Formulator", layout="wide")
st.title("AI-Powered TMR Formulator (Prototype)")

# --- Sidebar: Herd Inputs ---
st.sidebar.header("Herd & Cow Inputs")
bw_kg = st.sidebar.number_input("Cow Body Weight (kg)", value=650)
milk_kg = st.sidebar.number_input("Milk Yield (kg/d)", value=45)
dim = st.sidebar.number_input("Days in Milk (DIM)", value=75)
target_dmi = st.sidebar.number_input("Target DMI (kg DM/d)", value=31.0)

# --- Feed Library ---
st.sidebar.header("Feed Library (DM basis)")
default_feeds = [
    ["corn_silage", 0.08, 1.45, 8.5, 42.0, 30.0, 3.5, 20.0],
    ["alfalfa_haylage", 0.12, 1.55, 18.0, 38.0, 1.5, 2.5, 8.0],
    ["dry_hay", 0.10, 1.45, 16.0, 45.0, 1.0, 2.5, None],
    ["ground_corn", 0.20, 2.20, 8.5, 9.0, 72.0, 4.0, 8.0],
    ["sbm48", 0.35, 1.80, 48.0, 7.0, 1.0, 1.0, 4.0],
    ["ddgs", 0.18, 2.05, 30.0, 32.0, 2.0, 10.0, 4.0],
    ["whole_cottonseed", 0.22, 2.05, 23.0, 35.0, 0.0, 18.0, 3.0],
    ["rumen_protected_fat", 1.20, 6.5, 0.0, 0.0, 0.0, 98.0, 0.8],
]

feed_df = pd.DataFrame(default_feeds, columns=["Name","Price ($/kg DM)","NEL (Mcal/kg)","CP (%)","NDF (%)","Starch (%)","Fat (%)","Max kg DM"])
feed_df = st.sidebar.data_editor(feed_df, num_rows="dynamic")

@dataclass
class Feed:
    name: str
    price_per_kg_dm: float
    nel: float
    cp_pct: float
    ndf_pct: float
    starch_pct: float
    ee_pct: float
    max_kg_dm: float = None
    min_kg_dm: float = 0.0

feeds = []
for _, row in feed_df.iterrows():
    feeds.append(Feed(
        name=row["Name"],
        price_per_kg_dm=row["Price ($/kg DM)"],
        nel=row["NEL (Mcal/kg)"],
        cp_pct=row["CP (%)"],
        ndf_pct=row["NDF (%)"],
        starch_pct=row["Starch (%)"],
        ee_pct=row["Fat (%)"],
        max_kg_dm=row["Max kg DM"] if not pd.isna(row["Max kg DM"]) else None
    ))

# --- Placeholder NASEM requirements ---
def get_nasem_requirements(bw_kg, milk_kg, dim, dmi_kg):
    cp_pct_target = 17.0
    cp_g = dmi_kg * cp_pct_target / 100.0 * 1000.0
    nel_mcal = dmi_kg * 1.6
    return {
        "dmi_kg": dmi_kg,
        "nel_mcal": nel_mcal,
        "cp_g": cp_g,
        "ndf_min_pct": 26.0,
        "ndf_max_pct": 34.0,
        "starch_max_pct": 30.0
    }

requirements = get_nasem_requirements(bw_kg, milk_kg, dim, target_dmi)

# --- Formulate TMR ---
if st.button("Formulate TMR"):
    prob = pulp.LpProblem("TMR_Formulation", pulp.LpMinimize)
    x = {f.name: pulp.LpVariable(f"x_{f.name}", lowBound=f.min_kg_dm, upBound=f.max_kg_dm) for f in feeds}

    # Objective: minimize cost
    prob += pulp.lpSum([x[f.name]*f.price_per_kg_dm for f in feeds])

    # Constraints
    prob += pulp.lpSum([x[f.name] for f in feeds]) == requirements["dmi_kg"]
    prob += pulp.lpSum([x[f.name]*f.nel for f in feeds]) >= requirements["nel_mcal"]
    prob += pulp.lpSum([x[f.name]*(f.cp_pct/100)*1000 for f in feeds]) >= requirements["cp_g"]
    total_ndf_kg = pulp.lpSum([x[f.name]*(f.ndf_pct/100) for f in feeds])
    prob += total_ndf_kg >= requirements["ndf_min_pct"]/100*requirements["dmi_kg"]
    prob += total_ndf_kg <= requirements["ndf_max_pct"]/100*requirements["dmi_kg"]
    total_starch_kg = pulp.lpSum([x[f.name]*(f.starch_pct/100) for f in feeds])
    prob += total_starch_kg <= requirements["starch_max_pct"]/100*requirements["dmi_kg"]
    forage_vars = [x[f.name] for f in feeds if "hay" in f.name or "silage" in f.name]
    prob += pulp.lpSum(forage_vars) >= 0.20 * requirements["dmi_kg"]

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Results
    sol = {f.name: (x[f.name].varValue or 0.0) for f in feeds}
    total_dm = sum(sol.values())
    total_cp_g = sum(sol[f.name]*(f.cp_pct/100)*1000 for f in feeds)
    total_ndf_kg = sum(sol[f.name]*(f.ndf_pct/100) for f in feeds)
    total_starch_g = sum(sol[f.name]*(f.starch_pct/100)*1000 for f in feeds)
    total_ee_g = sum(sol[f.name]*(f.ee_pct/100)*1000 for f in feeds)
    total_nel = sum(sol[f.name]*f.nel for f in feeds)

    st.subheader("✅ Optimized TMR (kg DM per cow/day)")
    tmr_df = pd.DataFrame.from_dict(sol, orient='index', columns=['kg DM/d'])
    st.dataframe(tmr_df)

    st.subheader("📊 Diet Composition")
    st.markdown(f"""
    - Total DMI: **{total_dm:.2f} kg DM/d**  
    - NEL: **{total_nel:.2f} Mcal/d** | Diet NEL: **{total_nel/total_dm:.3f} Mcal/kg DM**  
    - CP: **{total_cp_g/1000:.2f} kg/d** | Diet CP: **{total_cp_g/total_dm/10:.2f} % DM**  
    - NDF: **{total_ndf_kg:.2f} kg/d** | Diet NDF: **{total_ndf_kg/total_dm*100:.2f} % DM**  
    - Starch: **{total_starch_g/1000:.2f} kg/d** | Diet Starch: **{total_starch_g/total_dm/10:.2f} % DM**  
    - Fat: **{total_ee_g/1000:.2f} kg/d** | Diet Fat: **{total_ee_g/total_dm/10:.2f} % DM**  
    - Estimated feed cost: **${pulp.value(prob.objective):.2f}/cow/day**
    """)
