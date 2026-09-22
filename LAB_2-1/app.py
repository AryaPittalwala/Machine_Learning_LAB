import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Boston Housing - Linear Regression", layout="wide")

FEATURES = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.title("🏠 Boston Housing — Linear Regression Explorer")
st.caption("Replicates the notebook: data loading, EDA, model training, evaluation, and prediction.")

uploaded = st.sidebar.file_uploader("Upload boston_housing_clean.csv", type=["csv"])
if uploaded is not None:
    ds = load_data(uploaded)
else:
    ds = load_data("boston_housing_clean.csv")
    st.sidebar.caption("Using bundled boston_housing_clean.csv (upload your own to replace it).")

# ---------------------------------------------------------------------------
# Sidebar controls (mirror train_test_split / model params from notebook)
# ---------------------------------------------------------------------------
st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# Train the model (cached on data + params)
@st.cache_resource
def train_model(df, test_size, random_state):
    x = df[FEATURES]
    y = df['MEDV']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return model, x, y, x_train, x_test, y_train, y_test, y_pred

model, x, y, x_train, x_test, y_train, y_test, y_pred = train_model(ds, test_size, random_state)

# ---------------------------------------------------------------------------
# Tabs mirroring the notebook sections
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Dataset", "📊 EDA", "🧮 Model & Coefficients", "📈 Evaluation", "🔮 Predict"]
)

# --- Tab 1: Load and understand the dataset ---
with tab1:
    st.subheader("Load and Understand the Dataset")

    st.write("**Shape:**", ds.shape)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Head**")
        st.dataframe(ds.head())
    with c2:
        st.write("**Tail**")
        st.dataframe(ds.tail())

    st.write("**Full Dataset**")
    st.dataframe(ds)

    st.write("**Describe (summary statistics)**")
    st.dataframe(ds.describe())

    st.write("**Info**")
    info_df = pd.DataFrame({
        "Column": ds.columns,
        "Non-Null Count": ds.notnull().sum().values,
        "Dtype": ds.dtypes.astype(str).values,
    })
    st.dataframe(info_df, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.write("**Null values per column**")
        st.dataframe(ds.isnull().sum().rename("null_count"))
    with c4:
        st.metric("Total null values", int(ds.isnull().sum().sum()))

# --- Tab 2: EDA ---
with tab2:
    st.subheader("Exploratory Data Analysis")

    st.write("**Distribution of MEDV**")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(ds['MEDV'], kde=True, bins=20, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.write("**Correlation Heatmap**")
    corr = ds.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Boston Housing Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

    st.write("**MEDV vs RM (scatter)**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=ds, x='RM', y='MEDV', ax=ax)
    ax.set_title('MEDV vs RM')
    ax.set_xlabel('RM (average number of rooms per dwelling)')
    ax.set_ylabel('MEDV (Median value of owner-occupied homes)')
    st.pyplot(fig)
    plt.close(fig)

    st.write("**MEDV vs RM with Regression Line**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=ds, x='RM', y='MEDV', scatter_kws={'alpha': 0.7}, ax=ax)
    ax.set_title('MEDV vs RM with Regression Line')
    ax.set_xlabel('RM')
    ax.set_ylabel('MEDV')
    st.pyplot(fig)
    plt.close(fig)

# --- Tab 3: Model & Coefficients ---
with tab3:
    st.subheader("Model Training")
    st.write(f"Features used: `{', '.join(FEATURES)}`")
    st.write(f"Train size: {x_train.shape[0]} rows | Test size: {x_test.shape[0]} rows")

    st.write("**Coefficients**")
    coef_df = pd.DataFrame({
        "Feature": x.columns,
        "Coefficient": np.round(model.coef_, 3)
    }).sort_values("Coefficient", ascending=False).reset_index(drop=True)
    st.dataframe(coef_df, use_container_width=True)
    st.write(f"**Intercept =** {round(model.intercept_, 3)}")

    strongest_pos = coef_df.iloc[0]
    strongest_neg = coef_df.iloc[-1]
    c1, c2 = st.columns(2)
    c1.success(f"Strongest positive: **{strongest_pos['Feature']}** = {strongest_pos['Coefficient']}")
    c2.error(f"Strongest negative: **{strongest_neg['Feature']}** = {strongest_neg['Coefficient']}")

    st.write("**Actual vs Predicted (test set)**")
    av_df = pd.DataFrame({"Actual": y_test.values, "Predicted": np.round(y_pred, 2)})
    st.dataframe(av_df, use_container_width=True, height=300)

# --- Tab 4: Evaluation ---
with tab4:
    st.subheader("Performance Metrics")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE", f"{mae:.3f}")
    m2.metric("MSE", f"{mse:.3f}")
    m3.metric("RMSE", f"{rmse:.3f}")
    m4.metric("R²", f"{r2:.3f}")
    m5.metric("Adjusted R²", f"{adjusted_r2:.3f}")

    st.info(
        f"The regression model explains **{r2*100:.0f}%** of the variation in house prices (MEDV). "
        f"The remaining **{(1-r2)*100:.0f}%** of the variation is not explained by the predictors in the model."
    )

    st.write("**Actual vs Predicted House Price**")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    ax.set_xlabel("Actual House Price")
    ax.set_ylabel("Predicted House Price")
    ax.set_title("Actual vs Predicted House Price")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.write("**Residual Plot**")
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title('Residual Plot')
    ax.set_xlabel('Predicted MEDV')
    ax.set_ylabel('Residuals')
    st.pyplot(fig)
    plt.close(fig)

# --- Tab 5: Predict ---
with tab5:
    st.subheader("Predict MEDV for Custom Input")
    st.write("Enter feature values (defaults are the dataset's median) to get a predicted house price.")

    cols = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(FEATURES):
        col = cols[i % 3]
        default = float(ds[feat].median())
        step = 0.01 if ds[feat].max() < 10 else 1.0
        input_vals[feat] = col.number_input(feat, value=default, step=step, format="%.4f")

    if st.button("Predict MEDV", type="primary"):
        input_df = pd.DataFrame([input_vals])[FEATURES]
        pred = model.predict(input_df)[0]
        st.success(f"Predicted MEDV (median home value): **${pred:.2f}k**")