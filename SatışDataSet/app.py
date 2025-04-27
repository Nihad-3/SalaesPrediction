import pickle
import numpy as np
import pandas as pd
import streamlit as st

# App Title
st.title("Nihad's Prediction App 🚀")

df = pd.read_csv('new_satis.csv')

with st.expander("Satış DataSeti"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="new_satis.csv",
        mime="text/csv"
    )


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
with st.sidebar:
    st.info('Məlumatları qeyd edin.')
    mehsul_sayi = st.select_slider('Məhsulun sayını seçin.',options=sorted(df['Məhsul sayi'].unique()), value=2)
    mehsul_nomresi = st.slider('Məhsulun nömrəsi seçin.', min_value=0, max_value=114, value=57)
    magaza = st.select_slider("📍Mağaza nömrəsini seçin.",options=sorted(df["Mağaza"].unique()),value=df['Mağaza'].median())
    mehsul_ceki = st.select_slider('Məhsulun çəkisi seçin.', options=sorted(df['Məhsul_çəki'].unique()), 
    value=df['Məhsul_çəki'].median())
    mehsul_ad = st.selectbox('Məhsulun adını seçin.',df['Məhsul_ad'].unique().tolist())
    meshul_cesidi = st.selectbox('Məhsulun çeşidini seçin.',df["Məhsulun_çeşidi"].unique().tolist())

input_data = {
    'Mağaza':magaza,
    'Məhsul_nomresi':mehsul_nomresi,
    'Məhsul sayi':mehsul_sayi,
    'Məhsul_ad':mehsul_ad,
    'Məhsul_çəki':mehsul_ceki,
    'Məhsulun_çeşidi':meshul_cesidi
}
# -----------
input_DataFrame = pd.DataFrame(input_data,index=[0])
st.write("📌**Daxil etdikləriniz:**", input_DataFrame)
if st.button("🔍 Nəticəni göstər."):
    prediction = model.predict(input_DataFrame)
    # Display result
    st.success(f"🎯 **Yekun Qiymət:** {round(prediction[0],2)}")



