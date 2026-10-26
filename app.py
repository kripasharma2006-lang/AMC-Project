from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from scipy.io import loadmat
import urllib.request

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="AMC Dashboard",
    page_icon="📡",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
MAT_PATH = BASE / "amc_dataset_improved.mat"
MODEL_DIR = BASE

CLASSES = ["AM", "FM", "BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]

FEATURES_DEFAULT = [
    "Mean Amplitude",
    "Standard Deviation of Amplitude",
    "Amplitude Kurtosis",
    "Standard Deviation of I",
    "Standard Deviation of Q",
    "Mean Absolute Phase Difference",
    "Standard Deviation of Phase Difference",
    "Mean Cosine of Phase Difference",
    "Mean Sine of Phase Difference",
    "Spectral Centroid",
    "Spectral Flatness",
    "Amplitude Skewness",
    "Peak-to-Average Power Ratio (PAPR)",
    "Phase Difference Kurtosis",
    "Instantaneous Frequency Standard Deviation",
    "Normalized Fourth-Order Cumulant"
]

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e8eef8;
}
[data-testid="stSidebar"] {
    background-color: #111c30;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #f1f6ff;
}
div[data-testid="stMetric"] {
    background: #14233a;
    border: 1px solid #263b58;
    padding: 14px;
    border-radius: 14px;
}
div[data-testid="stMetricValue"] {
    color: #65d9ff;
}
</style>
""", unsafe_allow_html=True)


def load_dataset():
    if not MAT_PATH.exists():
        return None, (
            f"Dataset not found: {MAT_PATH}\n"
            "Place amc_dataset_improved.mat in the same folder as app.py."
        )

    try:
        mat = loadmat(MAT_PATH)

        X = np.asarray(mat["X"], dtype=float)
        y = np.asarray(mat["y"]).reshape(-1).astype(int)

        if X.shape[0] != len(y) and X.shape[1] == len(y):
            X = X.T

        if X.shape[0] != len(y):
            raise ValueError(
                f"Feature shape {X.shape} does not match labels {len(y)}"
            )

        features = []
        raw_names = mat.get("featureNames", [])

        for item in np.asarray(raw_names, dtype=object).ravel():
            while isinstance(item, np.ndarray) and item.size == 1:
                item = item.item()

            if isinstance(item, bytes):
                item = item.decode(errors="ignore")

            if isinstance(item, str):
                features.append(item.strip())
            else:
                try:
                    name = "".join(
                        np.asarray(item).astype(str).ravel()
                    ).strip()
                    if name:
                        features.append(name)
                except Exception:
                    pass

        if len(features) != X.shape[1]:
            features = FEATURES_DEFAULT[:X.shape[1]]
            while len(features) < X.shape[1]:
                features.append(f"Feature {len(features) + 1}")

        # Correct dataset label mapping
        label_map = {
            1: "AM",
            2: "FM",
            3: "BPSK",
            4: "QPSK",
            5: "8PSK",
            6: "16QAM",
            7: "64QAM"
        }

        labels = np.array(
            [label_map.get(int(value), "Unknown") for value in y]
        )

        snr = np.asarray(
            mat.get(
                "snrColumn",
                np.full((len(y), 1), np.nan)
            )
        ).reshape(-1)

        if len(snr) != len(y):
            snr = np.full(len(y), np.nan)

        df = pd.DataFrame(X, columns=features)
        df["Modulation"] = pd.Categorical(
            labels,
            categories=CLASSES,
            ordered=True
        )
        df["SNR (dB)"] = snr

        return {
            "X": X,
            "y": y,
            "labels": labels,
            "snr": snr,
            "features": features,
            "df": df
        }, None

    except Exception as e:
        return None, f"Could not load dataset: {e}"



@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    loaded = {}
    errors = []

    # Download Random Forest model from Hugging Face
    rf_path = MODEL_DIR / "random_forest.pkl"
    rf_url = (
        "https://huggingface.co/kripa-sharma2006/"
        "amc-random-forest/resolve/main/random_forest.pkl"
    )

    if not rf_path.exists():
    try:
        with st.spinner("Downloading Random Forest model..."):
            temp_path = MODEL_DIR / "random_forest.tmp"
            urllib.request.urlretrieve(rf_url, str(temp_path))
            temp_path.replace(rf_path)
    except Exception as e:
        errors.append(f"Random Forest download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()

    model_files = [
        ("Random Forest", "random_forest.pkl"),
        ("SVM", "svm.pkl"),
        ("CNN", "cnn_model.keras"),
    ]

    for name, filename in model_files:
        path = MODEL_DIR / filename

        if not path.exists():
            errors.append(f"{filename} not found")
            continue

        try:
            if name == "CNN":
                from tensorflow.keras.models import load_model
                loaded[name] = load_model(path, compile=False)
            else:
                loaded[name] = joblib.load(path)
        except Exception as e:
            errors.append(f"{filename}: {e}")

    scaler = None
    scaler_path = MODEL_DIR / "scaler.pkl"

    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            errors.append(f"scaler.pkl: {e}")
    else:
        errors.append("scaler.pkl not found")

    return loaded, scaler, errors



def prepare_input(model_name, X, scaler):
    X = np.asarray(X, dtype=np.float32)

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    if model_name == "CNN":
        X = X.reshape(X.shape[0], X.shape[1], 1)

    return X


def predict(model_name, model, X, scaler):
    Xin = prepare_input(model_name, X, scaler)

    if model_name == "CNN":
        probabilities = np.asarray(
            model.predict(Xin, verbose=0)
        )

        if probabilities.ndim == 2 and probabilities.shape[1] == 7:
            index = int(np.argmax(probabilities[0]))
            return CLASSES[index], probabilities[0]

        return str(probabilities.squeeze()), None

    prediction = model.predict(Xin)[0]

    if isinstance(prediction, (str, np.str_)):
        label = str(prediction)
    else:
        number = int(prediction)

        if 1 <= number <= 7:
            label = CLASSES[number - 1]
        elif 0 <= number < 7:
            label = CLASSES[number]
        else:
            label = str(prediction)

    probabilities = None

    if hasattr(model, "predict_proba"):
        try:
            raw_probabilities = model.predict_proba(Xin)[0]
            probabilities = np.zeros(7)

            for i, cls in enumerate(
                getattr(model, "classes_", range(len(raw_probabilities)))
            ):
                number = int(cls)

                if 1 <= number <= 7:
                    probabilities[number - 1] = raw_probabilities[i]
                elif 0 <= number < 7:
                    probabilities[number] = raw_probabilities[i]

        except Exception:
            probabilities = None

    return label, probabilities


data, error = load_dataset()
models, scaler, model_errors = load_models()

st.sidebar.title("📡 AMC Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Signal Analysis",
        "Model Comparison",
        "Live Prediction",
        "Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("All 7 Modulation Classes")

for modulation in CLASSES:
    st.sidebar.write(f"• {modulation}")

st.title("📡 Automatic Modulation Classification")
st.caption(
    "Explore modulation features, SNR trends, model diagnostics, "
    "and predictions."
)

if error:
    st.error(error)
    st.stop()

df = data["df"]
X = data["X"]
labels = data["labels"]
snr = data["snr"]
features = data["features"]

counts = (
    pd.Series(labels)
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Samples", f"{len(df):,}")
    col2.metric("Features", X.shape[1])
    col3.metric("Modulation Classes", "7")

    valid_snr = snr[np.isfinite(snr)]

    col4.metric(
        "SNR Levels",
        len(np.unique(valid_snr)) if len(valid_snr) else "—"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Samples per Modulation")

        bar = px.bar(
            x=CLASSES,
            y=[int(counts[c]) for c in CLASSES],
            labels={
                "x": "Modulation",
                "y": "Number of Samples"
            },
            text=[int(counts[c]) for c in CLASSES]
        )

        bar.update_layout(template="plotly_dark")
        bar.update_traces(textposition="outside")

        st.plotly_chart(bar, use_container_width=True)

    with right:
        st.markdown("#### Class Distribution")

        pie = px.pie(
            names=CLASSES,
            values=[int(counts[c]) for c in CLASSES],
            hole=0.4
        )

        pie.update_layout(template="plotly_dark")

        st.plotly_chart(pie, use_container_width=True)

    st.markdown("#### Dataset Class Counts")

    st.dataframe(
        pd.DataFrame({
            "Modulation": CLASSES,
            "Samples": [int(counts[c]) for c in CLASSES]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "All classes are included: AM, FM, BPSK, QPSK, "
        "8PSK, 16QAM and 64QAM."
    )


# --------------------------------------------------
# PAGE 2: SIGNAL ANALYSIS
# --------------------------------------------------

elif page == "Signal Analysis":

    st.subheader("Signal Feature Analysis")

    chosen = st.selectbox(
        "Select Modulation",
        ["All classes"] + CLASSES
    )

    feature = st.selectbox(
        "Select Feature to Analyse",
        features
    )

    if chosen == "All classes":
        view = df
    else:
        view = df[df["Modulation"].astype(str) == chosen]

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            view,
            x=feature,
            color="Modulation",
            nbins=40,
            barmode="overlay",
            opacity=0.65
        )

        fig.update_layout(
            template="plotly_dark",
            title=f"Distribution: {feature}"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        trend_data = view[
            np.isfinite(view["SNR (dB)"])
        ].groupby(
            ["Modulation", "SNR (dB)"],
            observed=False
        )[feature].mean().reset_index()

        fig = px.line(
            trend_data,
            x="SNR (dB)",
            y=feature,
            color="Modulation",
            markers=True
        )

        # Correct layout code
        fig.update_layout(
            template="plotly_dark",
            title=f"{feature} vs SNR"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature Scatter Plot")

    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox(
            "X-axis Feature",
            features,
            index=0
        )

    with col2:
        y_feature = st.selectbox(
            "Y-axis Feature",
            features,
            index=min(1, len(features) - 1)
        )

    sample = (
        view.sample(min(4000, len(view)), random_state=42)
        if len(view)
        else view
    )

    fig = px.scatter(
        sample,
        x=x_feature,
        y=y_feature,
        color="Modulation",
        hover_data=["SNR (dB)"],
        opacity=0.7
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Feature Correlation Heatmap"):

        correlation = pd.DataFrame(
            X,
            columns=features
        ).corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )

        fig.update_layout(
            template="plotly_dark",
            height=650
        )

        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# PAGE 3: MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.subheader("Model Availability and Comparison")

    if model_errors:
        st.warning(
            "Some model files could not be loaded:\n\n"
            + "\n".join(f"- {e}" for e in model_errors)
        )

    if not models:
        st.info(
            "No trained models were loaded. "
            "Check the files inside the amc_models folder."
        )

    else:
        st.success(
            "Loaded models: " + ", ".join(models.keys())
        )

        maximum = min(2000, len(X))

        n = st.slider(
            "Number of Dataset Samples",
            min_value=min(100, maximum),
            max_value=maximum,
            value=min(500, maximum),
            step=100
        )

        indices = np.random.default_rng(42).choice(
            len(X),
            size=min(n, len(X)),
            replace=False
        )

        results = []

        for name, model in models.items():

            try:
                model_input = prepare_input(
                    name,
                    X[indices],
                    scaler
                )

                if name == "CNN":
                    predictions = np.argmax(
                        model.predict(model_input, verbose=0),
                        axis=1
                    )

                    predicted_names = [
                        CLASSES[int(p)] for p in predictions
                    ]

                else:
                    predictions = model.predict(model_input)
                    predicted_names = []

                    for prediction in predictions:

                        try:
                            number = int(prediction)

                            if 1 <= number <= 7:
                                predicted_names.append(
                                    CLASSES[number - 1]
                                )
                            elif 0 <= number < 7:
                                predicted_names.append(
                                    CLASSES[number]
                                )
                            else:
                                predicted_names.append(str(prediction))

                        except Exception:
                            predicted_names.append(str(prediction))

                accuracy = np.mean(
                    np.array(predicted_names) == labels[indices]
                )

                results.append({
                    "Model": name,
                    "Diagnostic Accuracy": float(accuracy)
                })

            except Exception as e:
                st.warning(
                    f"{name} could not be evaluated: {e}"
                )

        if results:

            result_df = pd.DataFrame(results)

            fig = px.bar(
                result_df,
                x="Model",
                y="Diagnostic Accuracy",
                text=result_df["Diagnostic Accuracy"].map(
                    lambda value: f"{value:.1%}"
                )
            )

            fig.update_layout(
                template="plotly_dark",
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "This is a diagnostic comparison on sampled dataset rows, "
                "not a separate held-out test-set evaluation."
            )

    st.markdown("#### Dataset Class Counts")

    st.dataframe(
        pd.DataFrame({
            "Modulation": CLASSES,
            "Samples": [int(counts[c]) for c in CLASSES]
        }),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# PAGE 4: LIVE PREDICTION
# --------------------------------------------------

elif page == "Live Prediction":

    st.subheader("Live Modulation Prediction")

    if not models:
        st.error(
            "No trained models were loaded. "
            "Check your amc_models folder."
        )
        st.stop()

    mode = st.radio(
        "Choose Input Method",
        [
            "Use a dataset sample",
            "Enter features manually"
        ],
        horizontal=True
    )

    submitted = True

    if mode == "Use a dataset sample":

        row = st.number_input(
            "Dataset Row Index (0-based)",
            min_value=0,
            max_value=len(X) - 1,
            value=0,
            step=1
        )

        row = int(row)
        model_input = X[row].reshape(1, -1)

        st.write(
            f"Actual Modulation: **{labels[row]}**"
        )

        if np.isfinite(snr[row]):
            st.write(f"SNR: **{snr[row]:g} dB**")

        with st.expander("View Feature Values"):

            st.dataframe(
                pd.DataFrame(
                    [model_input[0]],
                    columns=features
                ),
                use_container_width=True
            )

    else:

        defaults = np.nanmedian(X, axis=0)
        values = []

        with st.form("manual_features"):

            columns = st.columns(2)

            for i, feature_name in enumerate(features):

                with columns[i % 2]:

                    value = st.number_input(
                        feature_name,
                        value=float(defaults[i]),
                        format="%.6f",
                        key=f"manual_{i}"
                    )

                    values.append(value)

            submitted = st.form_submit_button(
                "Predict Modulation"
            )

        model_input = np.asarray(
            values,
            dtype=float
        ).reshape(1, -1)

    model_name = st.selectbox(
        "Select Prediction Model",
        list(models.keys())
    )

    if submitted:

        try:
            predicted, probabilities = predict(
                model_name,
                models[model_name],
                model_input,
                scaler
            )

            st.metric(
                "Predicted Modulation",
                predicted
            )

            if probabilities is not None and len(probabilities) == 7:

                fig = px.bar(
                    x=CLASSES,
                    y=probabilities,
                    labels={
                        "x": "Modulation",
                        "y": "Probability"
                    }
                )

                fig.update_layout(
                    template="plotly_dark",
                    yaxis_tickformat=".0%",
                    yaxis_range=[0, 1]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Prediction failed: {e}")

            st.caption(
                "Check that your trained model uses the same "
                "features and preprocessing scaler."
            )


# --------------------------------------------------
# PAGE 5: DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.subheader("Dataset Explorer")

    col1, col2 = st.columns(2)

    with col1:
        selected_classes = st.multiselect(
            "Select Modulation Classes",
            CLASSES,
            default=CLASSES
        )

    with col2:

        valid_snr = np.unique(
            snr[np.isfinite(snr)]
        )

        if len(valid_snr):
            snr_options = sorted(valid_snr.tolist())
        else:
            snr_options = [-10, -5, 0, 5, 10, 15, 20]

        selected_snr = st.multiselect(
            "Select SNR (dB)",
            snr_options,
            default=snr_options
        )

    filtered = df[
        df["Modulation"].astype(str).isin(selected_classes)
    ].copy()

    if (
        len(selected_snr)
        and np.isfinite(filtered["SNR (dB)"]).any()
    ):
        filtered = filtered[
            filtered["SNR (dB)"].isin(selected_snr)
        ]

    st.write(
        f"Rows matching filters: **{len(filtered):,}**"
    )

    st.dataframe(
        filtered.head(5000),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The table displays up to 5,000 rows. "
        "The CSV download contains all filtered rows."
    )

    st.download_button(
        "Download Filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "amc_filtered_dataset.csv",
        "text/csv"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "AMC Dashboard | Class mapping: "
    "1=AM, 2=FM, 3=BPSK, 4=QPSK, "
    "5=8PSK, 6=16QAM, 7=64QAM"
)
