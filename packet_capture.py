from scapy.all import sniff, IP, TCP, UDP
import joblib
import pandas as pd
import numpy as np

print("🚀 Loading trained model...")
model = joblib.load("rf_model.pkl")
print("✅ Model loaded!")

n_features = model.n_features_in_


def extract_features(packet):
    features = []

    # Basic packet features
    packet_length = len(packet)
    features.append(packet_length)

    if IP in packet:
        features.append(packet[IP].ttl)
        features.append(packet[IP].len)
        features.append(packet[IP].proto)
    else:
        features.extend([0, 0, 0])

    if TCP in packet:
        features.append(packet[TCP].sport)
        features.append(packet[TCP].dport)
        features.append(int(packet[TCP].flags))
    else:
        features.extend([0, 0, 0])

    if UDP in packet:
        features.append(packet[UDP].sport)
        features.append(packet[UDP].dport)
    else:
        features.extend([0, 0])

    # Fill remaining features with zeros
    while len(features) < n_features:
        features.append(0)

    return features[:n_features]


def process_packet(packet):
    features = extract_features(packet)

    feature_names = model.feature_names_in_
    df = pd.DataFrame([features], columns=feature_names)

    prediction = model.predict(df)

    if prediction[0] == 0:
        print("🟢 Normal Traffic")
    else:
        print("🔴 Intrusion Detected!")


print("🚀 Starting Real-Time Intrusion Detection...")
sniff(prn=process_packet, count=20)
print("✅ Detection Completed.")
