
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node, State

st.set_page_config(layout="wide")
st.title("Bayesian Network Modeling App (Pomegranate Version)")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    # Convert all columns to string (pomegranate requires discrete vars)
    data = data.dropna()
    for col in data.columns:
        data[col] = data[col].astype(str)

    # Learn structure and fit
    if st.button("Learn Bayesian Network"):
        model = BayesianNetwork.from_samples(data.values, algorithm='chow-liu', state_names=data.columns.tolist())
        st.success("Bayesian Network learned successfully!")

        # Draw graph
        edges = model.structure
        G = nx.DiGraph()
        for i, parents in enumerate(edges):
            for p in parents:
                G.add_edge(data.columns[p], data.columns[i])
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
        st.pyplot(plt)

        # Inference (basic)
        st.subheader("Query Probabilities with Evidence")
        evidence_vars = st.multiselect("Select evidence variables", data.columns.tolist())
        evidence = {}
        for var in evidence_vars:
            options = sorted(data[var].unique())
            val = st.selectbox(f"Select value for {var}", options, key=var)
            evidence[var] = val

        query_var = st.selectbox("Select a target variable", [col for col in data.columns if col not in evidence])
        if st.button("Run Inference"):
            # Estimate full distribution
            samples = pd.DataFrame(model.sample(n=5000), columns=data.columns)
            for ev, val in evidence.items():
                samples = samples[samples[ev] == val]
            result = samples[query_var].value_counts(normalize=True).round(3)
            st.write("Estimated probability distribution for:", query_var)
            st.bar_chart(result)
