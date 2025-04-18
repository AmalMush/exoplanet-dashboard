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
    df = pd.read_csv("cumulative.csv")  # Make sure this file is in the same folder
    df = df.dropna(subset=["koi_prad", "koi_insol", "koi_steff", "koi_teq"])
    df = df[df["koi_disposition"] == "CONFIRMED"]
    df = df.copy()
    df["habitability"] = (
        (df["koi_prad"].between(0.5, 2.0)) &
        (df["koi_insol"].between(0.75, 1.5)) &
        (df["koi_steff"].between(4000, 7000))
    ).astype(int)
    return df

df = load_data()
# st.write("DEBUG - Column Names in DataFrame:", df.columns.tolist())

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()


# Sidebar filters
st.sidebar.header("🌟 Filters")
min_radius, max_radius = st.sidebar.slider("Planet Radius (Earth radii)", 0.1, 20.0, (0.5, 2.0))
insol_range = st.sidebar.slider(
    "Insolation Flux (Earth = 1)",
    min_value=0.01,
    max_value=10000.0,
    value=(0.1, 5.0)
)
temp_range = st.sidebar.slider("Stellar Temperature (K)", 3000, 10000, (4000, 7000))
use_model = st.sidebar.checkbox("🤖 Use ML Model for Prediction")
koi_query = st.sidebar.text_input("🔍 Search KOI Planet (e.g. K00701.03)")


filtered_df = df[
    (df["koi_prad"].between(min_radius, max_radius)) &
    (df["koi_insol"].between(*insol_range)) &
    (df["koi_steff"].between(*temp_range))
]

# Filter by KOI or planet name if search is provided
if koi_query:
    filtered_df = filtered_df[
    filtered_df["kepoi_name"].str.contains(koi_query, case=False, na=False) |
    filtered_df["kepler_name"].str.contains(koi_query, case=False, na=False)
]
    ]

st.write(f"### 🔍 Showing {len(filtered_df)} planets matching filter criteria")
st.dataframe(filtered_df[["kepoi_name", "koi_prad", "koi_insol", "koi_teq", "koi_steff", "habitability"]])
# Convert filtered data to CSV
# Predict habitability using ML model or rule-based logic
features = ["koi_prad", "koi_period", "koi_srad", "koi_slogg", "koi_teq", "koi_insol", "koi_steff"]

if use_model:
    filtered_df = filtered_df.dropna(subset=features)
    filtered_df["habitability"] = model.predict(filtered_df[features])
else:
    filtered_df["habitability"] = (
        (filtered_df["koi_prad"].between(0.5, 2.0)) &
        (filtered_df["koi_insol"].between(0.75, 1.5)) &
        (filtered_df["koi_steff"].between(4000, 7000))
    ).astype(int)
if koi_query:
    koi_data = df[df["kepoi_name"].str.lower() == koi_query.lower()]
    
    if not koi_data.empty:
        st.markdown("---")
        st.subheader("🪐 KOI Planet Details")

        selected = koi_data.iloc[0]
        
        st.markdown(f"""
        **KOI Name:** `{selected['kepoi_name']}`  
        **Planet Radius:** {selected['koi_prad']} Earth radii  
        **Orbital Period:** {selected['koi_period']} days  
        **Star Radius:** {selected['koi_srad']} Solar radii  
        **Star Surface Gravity:** {selected['koi_slogg']}  
        **Equilibrium Temp:** {selected['koi_teq']} K  
        **Insolation Flux:** {selected['koi_insol']} Earth = 1  
        **Stellar Temp:** {selected['koi_steff']} K  
        """)

        if use_model and all(pd.notnull(selected[feature]) for feature in features):
            pred = model.predict([selected[features].values])[0]
            st.success(f"🌍 Predicted Habitability (ML): {'✅ Likely Habitable' if pred else '❌ Not Habitable'}")
        elif not use_model:
            is_habitable = (
                0.5 <= selected["koi_prad"] <= 2.0 and
                0.75 <= selected["koi_insol"] <= 1.5 and
                4000 <= selected["koi_steff"] <= 7000
            )
            st.info(f"🧪 Rule-Based Habitability: {'✅ Likely Habitable' if is_habitable else '❌ Not Habitable'}")
    else:
        st.warning("❌ KOI not found. Check the name and try again.")


csv = filtered_df.to_csv(index=False).encode('utf-8')

# Download button
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_planets.csv',
    mime='text/csv'
)


# Plot
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x="koi_insol", y="koi_prad", hue="habitability", palette="Set1", ax=ax)
ax.set_xscale('log')
ax.set_xlabel("Insolation Flux (log scale)")
ax.set_ylabel("Planet Radius (Earth radii)")
ax.set_title("Planet Radius vs. Insolation Flux")
st.pyplot(fig)


