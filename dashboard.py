import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Exoplanet Habitability Dashboard", layout="wide")
st.title("🪐 Exoplanet Habitability Explorer")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("cumulative.csv")
    df = df.dropna(subset=["pl_rade", "pl_insol", "st_teff", "pl_eqt"])
    df["habitability"] = (
        (df["pl_rade"].between(0.5, 2.0)) &
        (df["pl_insol"].between(0.75, 1.5)) &
        (df["st_teff"].between(4000, 7000))
    ).astype(int)
    return df

df = load_data()

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Sidebar filters
st.sidebar.header("🌟 Filters")
min_radius, max_radius = st.sidebar.slider("Planet Radius (Earth radii)", 0.1, 20.0, (0.5, 2.0))
insol_range = st.sidebar.slider("Insolation Flux (Earth = 1)", 0.01, 10000.0, (0.1, 5.0))
temp_range = st.sidebar.slider("Stellar Temperature (K)", 3000, 10000, (4000, 7000))
use_model = st.sidebar.checkbox("🤖 Use ML Model for Prediction")
planet_query = st.sidebar.text_input("🔍 Search Planet by Name (e.g. K2-18 b)")

# Apply filters
filtered_df = df[
    (df["pl_rade"].between(min_radius, max_radius)) &
    (df["pl_insol"].between(*insol_range)) &
    (df["st_teff"].between(*temp_range))
]

# Prediction Features used in model.pkl
features = ["pl_rade", "pl_eqt", "pl_insol", "st_teff"]

# ML Prediction with error handling
if use_model:
    filtered_df = filtered_df.dropna(subset=features)
    filtered_df[features] = filtered_df[features].apply(pd.to_numeric, errors='coerce')
    filtered_df = filtered_df.dropna()

    if not filtered_df.empty:
        filtered_df["habitability"] = model.predict(filtered_df[features])
    else:
        st.warning("⚠️ No data available after applying filters and cleaning. Try changing the sliders.")
else:
    filtered_df["habitability"] = (
        (filtered_df["pl_rade"].between(0.5, 3.0)) &
        (filtered_df["pl_insol"].between(0.75, 1.5)) &
        (filtered_df["st_teff"].between(3200, 7000))
    ).astype(int)

# Search by planet name
if planet_query:
    result_df = df[df["pl_name"].str.contains(planet_query, case=False, na=False)]
    if not result_df.empty:
        selected = result_df.iloc[0]
        st.markdown("---")
        st.subheader("🪐 Planet Details")
        st.markdown(f"""
        **Planet Name:** `{selected['pl_name']}`  
        **Radius:** {selected['pl_rade']} Earth radii  
        **Equilibrium Temperature:** {selected['pl_eqt']} K  
        **Insolation Flux:** {selected['pl_insol']} Earth = 1  
        **Stellar Temp:** {selected['st_teff']} K  
        **Stellar Radius:** {selected.get('st_rad', 'N/A')}  
        **Stellar Surface Gravity:** {selected.get('st_logg', 'N/A')}  
        """)
        
        if use_model and all(pd.notnull(selected[feature]) for feature in features):
            pred = model.predict([selected[features].values])[0]
            st.success(f"🌍 Predicted Habitability (ML): {'✅ Likely Habitable' if pred else '❌ Not Habitable'}")
        elif not use_model:
            is_hab = (
                0.5 <= selected["pl_rade"] <= 3.0 and
                0.75 <= selected["pl_insol"] <= 1.5 and
                3200 <= selected["st_teff"] <= 7000
            )
            st.info(f"🧪 Rule-Based Habitability: {'✅ Likely Habitable' if is_hab else '❌ Not Habitable'}")
    else:
        st.warning("❌ Planet not found. Check the name and try again.")

# Display table
st.write(f"### 🔍 Showing {len(filtered_df)} planets matching filter criteria")
st.dataframe(filtered_df[["pl_name", "pl_rade", "pl_insol", "pl_eqt", "st_teff", "habitability"]])

# Download CSV
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Data as CSV", data=csv, file_name='filtered_planets.csv', mime='text/csv')

# Plot
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x="pl_insol", y="pl_rade", hue="habitability", palette="Set1", ax=ax)
ax.set_xscale('log')
ax.set_xlabel("Insolation Flux (log scale)")
ax.set_ylabel("Planet Radius (Earth radii)")
ax.set_title("Planet Radius vs. Insolation Flux")
st.pyplot(fig)



"# force update" 
