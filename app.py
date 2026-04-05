"""Streamlit training dashboard for NNFS."""

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

import nnfs
from nnfs.layers import Dense, ReLU, Sigmoid
from nnfs.optim import SGD
from nnfs.utils import (
    accuracy_score, confusion_matrix_binary, f1_binary, 
    make_binary_classification, make_xor, train_test_split
)


st.set_page_config(page_title="NNFS Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .main-card {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(120,120,120,0.25);
        background: linear-gradient(135deg, rgba(49,130,206,0.08), rgba(128,90,213,0.08));
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 0.4rem 0.8rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(120,120,120,0.25);
        background: rgba(255,255,255,0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-card">
      <h2 style="margin-bottom:0.2rem;">NNFS Training Studio</h2>
      <p style="margin-top:0.1rem;">
        Interactive playground for training neural networks from scratch using the NNFS framework.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment Controls")
    dataset_name = st.selectbox("Dataset", ["XOR", "Binary Classification"])
    st.subheader("Model")
    hidden = st.slider("Hidden units", 4, 128, 16, 4)
    activation = st.selectbox("Activation", ["ReLU", "Sigmoid"])
    st.subheader("Training")
    lr = st.number_input("Learning rate", value=0.1, min_value=1e-4, max_value=1.0, format="%.4f")
    epochs = st.slider("Epochs", 50, 2000, 400, 50)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)
    seed = st.number_input("Random seed", value=42, min_value=0, max_value=100000)
    st.caption("Tip: Increase epochs for XOR and lower LR for smoother training.")
    train_btn = st.button("Train Model", use_container_width=True, type="primary")


def build_binary_model(input_dim: int, hidden: int, activation: str):
    """Build a binary classification model."""
    layers = [Dense(input_dim, hidden)]
    
    if activation == "ReLU":
        layers.append(ReLU())
    else:
        layers.append(Sigmoid())
    
    layers.extend([
        Dense(hidden, 1),
        Sigmoid()
    ])
    
    return nnfs.Sequential(*layers)


def decision_boundary_grid(model, X, resolution=100):
    """Generate decision boundary grid for visualization."""
    x_min, x_max = X.data[:, 0].min() - 0.5, X.data[:, 0].max() + 0.5
    y_min, y_max = X.data[:, 1].min() - 0.5, X.data[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model(nnfs.Tensor(mesh_points)).data.reshape(xx.shape)
    
    return xx, yy, Z


def _plot_decision_boundary(X, y, xx, yy, zz):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.contourf(xx, yy, zz, levels=20, alpha=0.55, cmap="coolwarm")
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y.reshape(-1),
        edgecolors="black",
        linewidths=0.35,
        s=28,
        cmap="coolwarm",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")
    return fig

if train_btn:
    with st.spinner("Training model..."):
        progress = st.progress(10, text="Preparing dataset...")
        np.random.seed(seed)
        if dataset_name == "XOR":
            X, y = make_xor(500, noise=0.1, random_state=seed)
        else:
            X, y = make_binary_classification(600, n_features=2, noise=0.2, random_state=seed)

        progress.progress(35, text="Building model...")
        model = build_binary_model(input_dim=X.shape[1], hidden=hidden, activation=activation)
        
        # Setup trainer
        optimizer = SGD(model.parameters(), lr=lr)
        loss_fn = nnfs.BCELoss()
        config = nnfs.TrainerConfig(epochs=epochs, batch_size=batch_size, log_interval=max(1, epochs // 10))
        trainer = nnfs.Trainer(model, loss_fn, optimizer, config)

        progress.progress(65, text="Running training loop...")
        history = trainer.fit(X, y)
        preds = model(X)
        progress.progress(100, text="Done.")

    st.success("Training completed successfully.")

    np.random.seed(seed)
    acc = accuracy_score(y, nnfs.Tensor((preds.data > 0.5).astype(int).flatten()))
    f1 = f1_binary(y, preds)
    cm = confusion_matrix_binary(y, nnfs.Tensor((preds.data > 0.5).astype(int).flatten()))
    final_loss = history.train_loss[-1] if history.train_loss else float("nan")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Accuracy", f"{acc:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1 Score", f"{f1:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Final Loss", f"{final_loss:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Training Curves", "Decision Boundary", "Confusion Matrix"])

    with tab1:
        left, right = st.columns(2)
        with left:
            st.subheader("Loss")
            st.line_chart({"loss": history.train_loss})
        with right:
            st.subheader("Accuracy")
            st.line_chart({"accuracy": history.train_accuracy})

    with tab2:
        if X.shape[1] == 2:
            xx, yy, zz = decision_boundary_grid(model, X)
            fig = _plot_decision_boundary(X.data, y.data, xx, yy, zz)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Decision boundary is available for 2D datasets only.")

    with tab3:
        cm_data = {
            "Pred 0": [int(cm[2]), int(cm[1])],
            "Pred 1": [int(cm[0]), int(cm[3])],
        }
        st.subheader("Confusion Matrix (Rows = True labels)")
        st.table(cm_data)
        st.caption(f"TN={cm[2]} FP={cm[1]} FN={cm[0]} TP={cm[3]}")
else:
    st.info("Configure an experiment from the sidebar and click **Train Model**.")
