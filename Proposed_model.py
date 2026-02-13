import numpy as np
from tensorflow.keras import layers, models
from confu_mat import *
from load_save import load

#  Real-Time Scoring & Alert Generation Layer

def risk_scoring_and_alerts(y_prob, high_th=0.75, medium_th=0.5):
    """
    Generate composite risk scores and alerts
    """
    risk_scores = y_prob.flatten()

    alerts = []
    for score in risk_scores:
        if score >= high_th:
            alerts.append("HIGH-RISK")
        elif score >= medium_th:
            alerts.append("MEDIUM-RISK")
        else:
            alerts.append("LOW-RISK")

    return risk_scores, alerts


def proposed(xtrain, ytrain, xtest, ytest):
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest  = xtest.reshape(xtest.shape[0],  xtest.shape[1],  1)

    timesteps = xtrain.shape[1]

    # 1. BiLSTM TEMPORAL FEATURE EXTRACTOR
    input_layer = layers.Input(shape=(timesteps, 1))

    bilstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(input_layer)

    bilstm = layers.Bidirectional(
        layers.LSTM(32)
    )(bilstm)

    # 2. LATENT SPACE LEARNING (Autoencoder-style)
    latent = layers.Dense(32, activation="relu", name="Latent_Space")(bilstm)

    decoder = layers.Dense(64, activation="relu")(latent)
    output_layer = layers.Dense(1, activation="sigmoid")(decoder)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        xtrain, ytrain,
        epochs=1,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Prediction
    y_prob = model.predict(xtest)
    ypred = (y_prob > 0.5).astype(int).flatten()

    # Metrics
    met = multi_confu_matrix(ytest, ypred)

    # 4. Real-Time Risk Scoring & Alerts
    risk_scores, alerts = risk_scoring_and_alerts(y_prob)

    return ypred, y_prob, risk_scores, alerts, met


