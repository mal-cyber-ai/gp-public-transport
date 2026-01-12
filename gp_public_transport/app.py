import streamlit as st
import matplotlib.pyplot as plt
from gp_transport import run_gp

st.title("Public Transport Optimization using Genetic Programming")

st.write("This application demonstrates Genetic Programming applied to public transport data.")

if st.button("Run Genetic Programming"):
    best, log = run_gp()
    st.subheader("Best GP Model")
    st.code(str(best))

    gens = log.select("gen")
    mins = log.select("min")

    fig, ax = plt.subplots()
    ax.plot(gens, mins)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Minimum MSE")
    ax.set_title("GP Convergence Curve")
    st.pyplot(fig)
