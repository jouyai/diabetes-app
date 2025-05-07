import streamlit as st
import numpy as np
import joblib

# Load model dan preprocessing
model = joblib.load("model/diabetes_model.pkl")
scaler = joblib.load("model/scaler.pkl")
imputer = joblib.load("model/imputer.pkl")

st.title("Prediksi Diabetes 🔍")
st.write("Masukkan data pasien untuk memprediksi apakah berisiko diabetes.")

# Input dari user
pregnancies = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0)
blood_pressure = st.number_input("Blood Pressure", 0)
skin_thickness = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

# Tombol prediksi
if st.button("Prediksi"):
    # Data user → array
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, diabetes_pedigree, age]])

    # Imputasi dan scaling
    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)

    # Buat fitur tambahan
    glucose_insulin_ratio = data_scaled[0][1] / (data_scaled[0][3] + 1)
    bmi_age = data_scaled[0][5] * data_scaled[0][7]
    glucose_preg = data_scaled[0][1] * data_scaled[0][0]
    bp_age = data_scaled[0][2] * data_scaled[0][7]

    # Gabungkan fitur asli + fitur tambahan
    final_data = np.hstack([data_scaled, [[glucose_insulin_ratio, bmi_age, glucose_preg, bp_age]]])

    # Prediksi
    prediction = model.predict(final_data)[0]
    result = "POSITIF Diabetes ❗" if prediction == 1 else "Negatif Diabetes ✅"
    st.subheader(result)

    # Menambahkan tips
    if prediction == 1:
        st.write("""
        **Tips untuk menghindari risiko diabetes**:
        - **Makan makanan sehat**: Pilih makanan dengan indeks glikemik rendah, seperti sayuran, buah-buahan, dan biji-bijian utuh.
        - **Olahraga secara teratur**: Berusaha untuk berolahraga setidaknya 30 menit setiap hari untuk menjaga berat badan dan meningkatkan sensitivitas insulin.
        - **Jaga berat badan yang sehat**: Menurunkan berat badan, jika diperlukan, dapat membantu mengurangi risiko diabetes.
        - **Pantau kadar gula darah**: Rutin memeriksa kadar gula darah jika Anda berisiko tinggi.
        - **Hindari stres**: Stres berlebihan dapat memengaruhi kadar gula darah Anda, jadi penting untuk memiliki kebiasaan manajemen stres.
        - **Tidur yang cukup**: Tidur yang cukup sangat penting untuk menjaga keseimbangan hormon dan metabolisme tubuh.

        Jaga kesehatan Anda dan lakukan langkah pencegahan untuk hidup lebih sehat!
        """)
    else:
        st.write("""
        **Tips untuk menjaga kesehatan dan mencegah diabetes**:
        - **Pertahankan pola makan seimbang**: Perbanyak konsumsi buah, sayuran, dan protein sehat.
        - **Olahraga teratur**: Lakukan aktivitas fisik seperti jalan kaki, bersepeda, atau latihan kekuatan.
        - **Jaga berat badan ideal**: Menjaga berat badan yang sehat membantu tubuh mengelola gula darah.
        - **Rutin cek kesehatan**: Periksa kadar gula darah Anda secara rutin untuk pemantauan lebih lanjut.
        - **Kurangi konsumsi gula dan karbohidrat olahan**: Pilih sumber karbohidrat yang lebih sehat dan tidak mengandung banyak gula tambahan.
        - **Kelola stres**: Manajemen stres yang baik berkontribusi pada kesehatan secara keseluruhan.
        """)