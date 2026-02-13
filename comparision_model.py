from tensorflow.keras import layers, models
from confu_mat import *
from load_save import load

def risk_scoring_and_alerts(y_prob, high_th=0.75, medium_th=0.5):
    risk_scores = y_prob.flatten()
    alerts = []

    for s in risk_scores:
        if s >= high_th:
            alerts.append("HIGH-RISK")
        elif s >= medium_th:
            alerts.append("MEDIUM-RISK")
        else:
            alerts.append("LOW-RISK")

    return risk_scores, alerts


def build_model(timesteps):
    inp = layers.Input(shape=(timesteps, 1))

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(32))(x)

    latent = layers.Dense(32, activation="relu", name="Latent_Space")(x)

    x = layers.Dense(64, activation="relu")(latent)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def federated_avg(weights_list):
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights



def Fed_SCR(xtrain, ytrain, xtest, ytest,
                       num_clients=5, rounds=5, local_epochs=1):

    # Reshape
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest  = xtest.reshape(xtest.shape[0],  xtest.shape[1],  1)

    timesteps = xtrain.shape[1]

    # Split data across clients
    client_x = np.array_split(xtrain, num_clients)
    client_y = np.array_split(ytrain, num_clients)

    # Global model
    global_model = build_model(timesteps)

    for r in range(rounds):
        client_weights = []

        for c in range(num_clients):
            local_model = build_model(timesteps)
            local_model.set_weights(global_model.get_weights())

            local_model.fit(
                client_x[c], client_y[c],
                epochs=local_epochs,
                batch_size=128,
                verbose=0
            )

            client_weights.append(local_model.get_weights())

        # FedAvg
        global_model.set_weights(federated_avg(client_weights))

    # Prediction
    y_prob = global_model.predict(xtest)
    ypred = (y_prob > 0.5).astype(int).flatten()

    # Metrics
    met = multi_confu_matrix(ytest, ypred)

    # Risk Scoring
    risk_scores, alerts = risk_scoring_and_alerts(y_prob)

    return ypred, y_prob, risk_scores, alerts, met

#  Real-Time Scoring & Alert Generation Layer

def risk_scoring_and_alerts(y_prob, high_th=0.75, medium_th=0.5):
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



def LSTM_MPC(xtrain, ytrain, xtest, ytest):

    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest  = xtest.reshape(xtest.shape[0],  xtest.shape[1],  1)

    timesteps = xtrain.shape[1]

    # 1. LSTM TEMPORAL FEATURE EXTRACTOR
    input_layer = layers.Input(shape=(timesteps, 1))

    lstm = layers.LSTM(64, return_sequences=True)(input_layer)
    lstm = layers.LSTM(32)(lstm)

    # 2. LATENT SPACE LEARNING
    latent = layers.Dense(32, activation="relu", name="Latent_Space")(lstm)

    decoder = layers.Dense(64, activation="relu")(latent)
    output_layer = layers.Dense(1, activation="sigmoid")(decoder)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(xtrain, ytrain,epochs=1,batch_size=128,validation_split=0.1,verbose=1)
    # Prediction
    y_prob = model.predict(xtest)
    ypred = (y_prob > 0.5).astype(int).flatten()
    # Metrics
    met = multi_confu_matrix(ytest, ypred)
    # Risk Scoring
    risk_scores, alerts = risk_scoring_and_alerts(y_prob)
    return ypred, y_prob, risk_scores, alerts, met


def risk_scoring_and_alerts(y_prob, high_th=0.75, medium_th=0.5):
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



def ADLA_FL(xtrain, ytrain, xtest, ytest):

    # Reshape input
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest  = xtest.reshape(xtest.shape[0],  xtest.shape[1],  1)
    timesteps = xtrain.shape[1]
    # Encoder
    input_layer = layers.Input(shape=(timesteps, 1))
    enc = layers.LSTM(64, return_sequences=True)(input_layer)
    enc = layers.LSTM(32)(enc)
    latent = layers.Dense(16, activation="relu", name="Latent_Space")(enc)
    # Decoder (Autoencoder)
    dec = layers.RepeatVector(timesteps)(latent)
    dec = layers.LSTM(32, return_sequences=True)(dec)
    dec = layers.LSTM(64, return_sequences=True)(dec)
    reconstructed = layers.TimeDistributed(
        layers.Dense(1), name="Reconstruction"
    )(dec)
    # Classifier Head

    clf = layers.Dense(32, activation="relu")(latent)
    output_layer = layers.Dense(1, activation="sigmoid", name="Classifier")(clf)

    model = models.Model(inputs=input_layer,outputs=[output_layer, reconstructed])
    model.compile(optimizer="adam",loss={
            "Classifier": "binary_crossentropy",
            "Reconstruction": "mse"
        },
        loss_weights={
            "Classifier": 1.0,
            "Reconstruction": 0.5
        },
        metrics={"Classifier": "accuracy"}
    )

    model.fit(xtrain,
        {
            "Classifier": ytrain,
            "Reconstruction": xtrain
        },
        epochs=1,batch_size=128,validation_split=0.1,verbose=1)
    y_prob, _ = model.predict(xtest)
    ypred = (y_prob > 0.5).astype(int).flatten()
    met = multi_confu_matrix(ytest, ypred)
    risk_scores, alerts = risk_scoring_and_alerts(y_prob)
    return ypred, y_prob, risk_scores, alerts, met



#  Real-Time Scoring & Alert Generation Layer
def risk_scoring_and_alerts(y_prob, high_th=0.75, medium_th=0.5):
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

# Residual Block
def res_block(x, filters, kernel_size=3):
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

# ResBlock + CNN
def Res_block_CNN(xtrain, ytrain, xtest, ytest):

    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    xtest  = xtest.reshape(xtest.shape[0],  xtest.shape[1],  1)

    timesteps = xtrain.shape[1]

    input_layer = layers.Input(shape=(timesteps, 1))

    x = layers.Conv1D(32, 3, padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = res_block(x, 32)
    x = res_block(x, 64)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = res_block(x, 128)
    x = layers.GlobalAveragePooling1D()(x)

    latent = layers.Dense(32, activation="relu", name="Latent_Space")(x)

    x = layers.Dense(64, activation="relu")(latent)
    output_layer = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    model.fit(xtrain, ytrain,epochs=1,batch_size=128,validation_split=0.1,verbose=1)
    y_prob = model.predict(xtest)
    ypred = (y_prob > 0.5).astype(int).flatten()
    met = multi_confu_matrix(ytest, ypred)
    risk_scores, alerts = risk_scoring_and_alerts(y_prob)
    return ypred, y_prob, risk_scores, alerts, met
