# ============================================================
# REAL-TIME INTRUSION DETECTION SYSTEM
# Final Year Project
# ============================================================

# ==============================
# IMPORT LIBRARIES
# ==============================

import os
import datetime
import io

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scapy.all import sniff
from packet_capture import extract_features

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

os.environ["SCAPY_CACHE"] = "0"

# ============================================================
# STREAMLIT PAGE TITLE
# ============================================================

st.markdown(
    """
<div style='text-align:center;margin-bottom:30px'>
    <h1 style='color:#38bdf8'>🛡️ Real-Time Intrusion Detection System</h1>
    <p style='color:#9ca3af;font-size:18px'>
    AI Powered Network Security Monitoring Dashboard
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CUSTOM UI DESIGN
# ============================================================

st.markdown(
    """
<style>

/* Main background */
.stApp {
    background-color: #0f172a;
}

/* Title */
h1 {
    color: #38bdf8;
    text-align: center;
}

/* Section headers */
h2, h3 {
    color: #e2e8f0;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #334155;
}

/* Metric label */
[data-testid="metric-container"] label {
    color: #cbd5f5 !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Alert box */
.stAlert {
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("rf_model.pkl")

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "attack_count" not in st.session_state:
    st.session_state.attack_count = 0

if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0

if "logs" not in st.session_state:
    st.session_state.logs = []

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False


# ============================================================
# PACKET PROCESSING FUNCTION
# ============================================================


def process_packet(packet):

    features = extract_features(packet)
    feature_names = model.feature_names_in_

    df = pd.DataFrame([features], columns=feature_names)

    prediction = model.predict(df)

    from scapy.layers.inet import IP

    src_ip = packet[IP].src if packet.haslayer(IP) else "Unknown"

    if prediction[0] == 0:
        st.session_state.normal_count += 1
        traffic_type = "Normal"
    else:
        st.session_state.attack_count += 1
        traffic_type = "Intrusion"

    st.session_state.logs.append(
        {
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Source IP": src_ip,
            "Traffic Type": traffic_type,
        }
    )


# ============================================================
# MONITORING CONTROL
# ============================================================

st.markdown("### 🔍 Network Traffic Monitoring")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Monitoring"):
        st.session_state.monitoring = True

with col2:
    if st.button("⏹ Stop Monitoring"):
        st.session_state.monitoring = False

if st.session_state.monitoring:
    sniff(prn=process_packet, count=20)
    st.rerun()


# ============================================================
# LIVE TRAFFIC METRICS
# ============================================================

total_packets = st.session_state.attack_count + st.session_state.normal_count

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🟢 Normal Traffic", st.session_state.normal_count)

with col2:
    st.metric("🔴 Intrusions Detected", st.session_state.attack_count)

with col3:
    st.metric("📦 Total Packets", total_packets)


# ============================================================
# ALERT STATUS
# ============================================================

if st.session_state.attack_count > st.session_state.normal_count:
    st.error("🔴 ALERT: System Under Possible Attack!")
else:
    st.success("🟢 System Secure - Normal Traffic Dominant")


# ============================================================
# THREAT ANALYSIS
# ============================================================

if total_packets > 0:
    attack_percentage = (st.session_state.attack_count / total_packets) * 100
else:
    attack_percentage = 0

st.markdown(
    """
<div style="background-color:#1e293b;padding:15px;border-radius:10px">
<h3>📊 Threat Intelligence Analysis</h3>
</div>
""",
    unsafe_allow_html=True,
)

st.write(f"Attack Percentage: {attack_percentage:.2f}%")

if attack_percentage > 50:
    st.error("🚨 High Risk Level")
elif attack_percentage > 20:
    st.warning("⚠ Medium Risk")
else:
    st.success("✅ Low Risk")


# ============================================================
# TRAFFIC DISTRIBUTION CHART
# ============================================================

st.markdown("### 📈 Traffic Visualization")

if total_packets > 0:
    fig, ax = plt.subplots()

    ax.pie(
        [st.session_state.normal_count, st.session_state.attack_count],
        labels=["Normal Traffic", "Intrusion"],
        autopct="%1.1f%%",
        colors=["green", "red"],
    )

    ax.set_title("Traffic Distribution")

    st.pyplot(fig)


# ============================================================
# INTRUSION LOG TABLE
# ============================================================

st.markdown("### 📜 Real-Time Intrusion Log")

if st.session_state.logs:
    log_df = pd.DataFrame(st.session_state.logs)
    st.dataframe(log_df)
else:
    st.write("No traffic recorded yet.")


# ============================================================
# PDF REPORT GENERATION
# ============================================================


def generate_pdf():

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    elements = []
    styles = getSampleStyleSheet()

    total_packets = st.session_state.attack_count + st.session_state.normal_count

    attack_percentage = (
        st.session_state.attack_count / total_packets * 100 if total_packets > 0 else 0
    )

    data = [
        ["Total Packets", total_packets],
        ["Normal Traffic", st.session_state.normal_count],
        ["Intrusions Detected", st.session_state.attack_count],
        ["Attack Percentage", f"{attack_percentage:.2f}%"],
    ]

    elements.append(Paragraph("Intrusion Detection System Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    table = Table(data)

    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )

    elements.append(table)

    doc.build(elements)

    buffer.seek(0)

    return buffer


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.markdown("### 📄 Generate Security Report")

if total_packets > 0:
    pdf_file = generate_pdf()

    st.download_button(
        label="Download IDS Report (PDF)",
        data=pdf_file,
        file_name="IDS_Report.pdf",
        mime="application/pdf",
    )

else:
    st.write("No data available to generate report.")


# ============================================================
# MODEL PERFORMANCE (LAST SECTION)
# ============================================================

st.subheader("📊 Model Evaluation")

data = pd.read_csv("traffic_prp_200.csv")

encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = encoder.fit_transform(data[col])

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

st.subheader("Classification Report")

report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
).transpose()

st.dataframe(report_df)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"],
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
