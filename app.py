import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# ======================================================
# SAFE R² FUNCTION
# ======================================================
def safe_r2_score(y_true, y_pred):
    if len(y_true) < 2:
        return None
    return r2_score(y_true, y_pred)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Regression Explorer", layout="wide")

st.title("📈 Regression Explorer (MLE & MAP – OLS, Ridge, LASSO, Elastic Net)")
st.markdown("""
This project integrates **statistical estimation theory (MLE & MAP)**
with **modern regression models** using interactive visualizations,
numerical evaluation, and residual diagnostics.
""")

# ======================================================
# 1. DATA OPTIONS
# ======================================================
st.sidebar.header("1. Data Options")
source = st.sidebar.radio("Choose data source", ["Sample: House Price", "Upload CSV"])

if source == "Sample: House Price":
    df = pd.read_csv("data/house_price.csv")
else:
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.stop()
    df = pd.read_csv(file)

df = df.select_dtypes(include=np.number)

st.subheader("Dataset Preview")
st.dataframe(df.head())

if df.shape[1] < 2:
    st.error("Dataset must contain at least 2 numeric columns.")
    st.stop()

# ======================================================
# 2. FEATURES & TARGET
# ======================================================
st.sidebar.header("2. Features & Target")

target = st.sidebar.selectbox("Select target (y)", df.columns, index=len(df.columns) - 1)
features = st.sidebar.multiselect(
    "Select feature columns (X)",
    [c for c in df.columns if c != target],
    default=[c for c in df.columns if c != target][:2]
)

if len(features) == 0:
    st.warning("Select at least one feature.")
    st.stop()

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random state", 0, 100, 42)

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

if len(y_test) < 2:
    st.info("ℹ️ R² requires at least 2 test samples. Increase test size if needed.")

# ======================================================
# 3. TRAIN REGRESSION MODELS
# ======================================================
st.sidebar.header("3. Regression Hyperparameters")

alpha = st.sidebar.slider("Alpha (λ)", 0.01, 10.0, 1.0)
l1_ratio = st.sidebar.slider("Elastic Net l1_ratio", 0.0, 1.0, 0.5)

models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=alpha),
    "Lasso": Lasso(alpha=alpha),
    "Elastic Net": ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
}

trained_models = {}
metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    y_pred = model.predict(X_test)

    r2 = safe_r2_score(y_test, y_pred)

    metrics.append({
        "Model": name,
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": "Not defined" if r2 is None else round(r2, 3)
    })

metrics_df = pd.DataFrame(metrics)

best_model = metrics_df.sort_values("MSE").iloc[0]
st.success(
    f"✅ Recommended Model: {best_model['Model']} "
    f"(Lowest MSE = {best_model['MSE']:.3f})"
)

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & Correlation",
    "📐 Theory & Formulas",
    "📋 Numerical Solution",
    "📈 Plots (2D & 3D)",
    "📌 Likelihood & Priors",
    "🔔 Residuals & Diagnostics"
])

# ------------------------------------------------------
# TAB 1: DATA & CORRELATION
# ------------------------------------------------------
with tab1:
    st.write(df.describe())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ------------------------------------------------------
# TAB 2: THEORY & FORMULAS (RAW STRING FIXED)
# ------------------------------------------------------
with tab2:
    st.markdown(r"""
## Linear Regression Model
\[
y = X\beta + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
\]

## MLE – Ordinary Least Squares
\[
\hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty
\]

## MAP – Ridge Regression (Gaussian Prior)
\[
\hat{\beta}_{Ridge} =
\arg\min_\beta
\left(\|y - X\beta\|^2 + \lambda \|\beta\|_2^2\right)
\]

## MAP – LASSO Regression (Laplace Prior)
\[
\hat{\beta}_{LASSO} =
\arg\min_\beta
\left(\|y - X\beta\|^2 + \lambda \|\beta\|_1\right)
\]

## Elastic Net
\[
\hat{\beta} =
\arg\min_\beta
\left(
\|y - X\beta\|^2 +
\lambda(\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2)
\right)
\]

### Interpretation
- OLS → MLE  
- Ridge → MAP with Gaussian prior  
- LASSO → MAP with Laplace prior  
- Elastic Net → MAP with mixed priors  
""")

# ------------------------------------------------------
# TAB 3: NUMERICAL SOLUTION
# ------------------------------------------------------
with tab3:
    st.subheader("Model Performance")
    st.dataframe(metrics_df)

    st.subheader("Coefficient Shrinkage Comparison")
    coef_df = pd.DataFrame({
        "Feature": features,
        "OLS": trained_models["OLS"].coef_,
        "Ridge": trained_models["Ridge"].coef_,
        "Lasso": trained_models["Lasso"].coef_
    })
    st.dataframe(coef_df)

# ------------------------------------------------------
# TAB 4: ADVANCED 3D VISUALIZATION
# ------------------------------------------------------
with tab4:
    st.subheader("Actual vs Predicted (OLS)")
    st.line_chart(pd.DataFrame({
        "Actual": y_test,
        "Predicted": trained_models["OLS"].predict(X_test)
    }))

    if len(features) >= 2:
        st.subheader("Interactive 3D Data Points")
        fig3d = px.scatter_3d(
            df,
            x=features[0],
            y=features[1],
            z=target,
            color=target,
            opacity=0.8
        )
        st.plotly_chart(fig3d, width="stretch")

        st.subheader("Advanced 3D Regression Plane (All Models)")

        plane_model_name = st.selectbox(
            "Choose model for 3D regression plane",
            list(trained_models.keys())
        )
        plane_model = trained_models[plane_model_name]

        f1, f2 = features[0], features[1]
        gx, gy = np.meshgrid(
            np.linspace(df[f1].min(), df[f1].max(), 25),
            np.linspace(df[f2].min(), df[f2].max(), 25)
        )

        grid_points = np.c_[gx.ravel(), gy.ravel()]

        if len(features) > 2:
            extra = df[features[2:]].mean().values
            grid_points = np.hstack([
                grid_points,
                np.tile(extra, (grid_points.shape[0], 1))
            ])

        gz = plane_model.predict(grid_points).reshape(gx.shape)

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(gx, gy, gz, alpha=0.55, cmap="viridis")
        ax.scatter(df[f1], df[f2], df[target], color="red", s=40)

        fitted = plane_model.predict(X_train)
        for i in range(len(X_train)):
            ax.plot(
                [X_train[i, 0], X_train[i, 0]],
                [X_train[i, 1], X_train[i, 1]],
                [y_train[i], fitted[i]],
                color="black",
                alpha=0.25
            )

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_zlabel(target)
        ax.set_title(f"3D Regression Plane ({plane_model_name})")

        st.pyplot(fig)

# ------------------------------------------------------
# TAB 5: LIKELIHOOD & PRIORS
# ------------------------------------------------------
with tab5:
    st.markdown(r"""
\[
MAP = \arg\max (\log p(y|X,\beta) + \log p(\beta))
\]
""")

# ------------------------------------------------------
# TAB 6: RESIDUAL ANALYSIS
# ------------------------------------------------------
with tab6:
    model_name = st.selectbox(
        "Choose model for residual analysis",
        list(trained_models.keys())
    )

    model = trained_models[model_name]
    fitted = model.predict(X_train)
    residuals = y_train - fitted

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title("Histogram of Residuals (Gaussian)")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.scatter(fitted, residuals, alpha=0.7)
        ax2.axhline(0, linestyle="--")
        ax2.set_title("Residuals vs Fitted")
        st.pyplot(fig2)

    st.markdown("Gaussian and centered residuals indicate a good linear model fit.")

st.success("🚀 Advanced Regression Explorer executed successfully!")
