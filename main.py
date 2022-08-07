"""Main."""

from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

from face_detect import create_face_area_data
from train import train_model, save_model


df = create_face_area_data()

model = train_model(df)

df["predicted_distance (m)"] = model.predict(
    df["face area (pixel)"].values.reshape(-1, 1)
)

st.dataframe(df)

fig, ax = plt.subplots()
ax.scatter(x=df["distance (m)"], y=df["face area (pixel)"])
ax.set_xlabel("distance (m)")
ax.set_ylabel("face area (pixel)")
st.pyplot(fig)

save_model(model)
