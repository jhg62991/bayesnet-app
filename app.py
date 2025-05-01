
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

st.title("Bayesian Network Modeling App")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    if st.button("Learn Network Structure"):
        hc = HillClimbSearch(data, scoring_method=BicScore(data))
        model = hc.estimate()
        bn = BayesianNetwork(model.edges())
        bn.fit(data, estimator=BayesianEstimator)

        st.write("Learned Network Edges:", model.edges())

        # Draw graph
        nx_graph = nx.DiGraph(model.edges())
        pos = nx.spring_layout(nx_graph)
        plt.figure(figsize=(10,6))
        nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
        st.pyplot(plt)

        # Inference section
        st.subheader("Run Inference")
        infer = VariableElimination(bn)
        query_var = st.selectbox("Select variable to predict", data.columns)
        evidence_vars = st.multiselect("Select evidence variables", [col for col in data.columns if col != query_var])

        evidence = {}
        for ev in evidence_vars:
            options = data[ev].dropna().unique().tolist()
            selected = st.selectbox(f"Value of {ev}", options)
            evidence[ev] = selected

        if st.button("Compute Inference"):
            result = infer.query([query_var], evidence=evidence)
            st.write(result)
