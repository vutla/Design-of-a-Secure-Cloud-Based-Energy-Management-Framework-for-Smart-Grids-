import numpy as np
from confu_mat import *
from plot_res import *
from Data_gen import *
from Proposed_model import *
from comparision_model import *

def full_analysis():
    Data_gen()
    X_train = load("X_train")
    X_test = load("X_test")
    y_train = load("y_train")
    y_test = load("y_test")

    #Proposed BTEA
    ypred, y_prob, risk_scores, alerts, met = proposed(X_train, y_train, X_test, y_test)

    for i in range(10):
        print(f"{i + 1} | Risk Score: {risk_scores[i]:.3f} | Alert: {alerts[i]}")

    # Comparision_model

    # Fed-SCR
    X_train = load("X_train")
    X_test = load("X_test")
    y_train = load("y_train")
    y_test = load("y_test")

    ypred, y_prob, risk_scores, alerts, met = Fed_SCR(X_train, y_train, X_test, y_test)
    save('Fed-SCR_MET',met)
    for i in range(10):
        print(f"{i + 1} | Risk Score: {risk_scores[i]:.3f} | Alert: {alerts[i]}")

    # LSTM-MPC
    ypred, y_prob, risk_scores, alerts, met =LSTM_MPC(X_train, y_train, X_test, y_test)
    save('LSTM_MPC_MET', met)
    for i in range(10):
        print(f"{i + 1} | Risk Score: {risk_scores[i]:.3f} | Alert: {alerts[i]}")

    # ADLA-FL
    ypred, y_prob, risk_scores, alerts, met =ADLA_FL(X_train, y_train, X_test, y_test)
    save('ADLA_FL_MET', met)
    for i in range(10):
        print(f"{i + 1} | Risk Score: {risk_scores[i]:.3f} | Alert: {alerts[i]}")

    # Res-block+CNN
    ypred, y_prob, risk_scores, alerts, met =Res_block_CNN(X_train, y_train, X_test, y_test)
    save('Res_block_CNN_MET', met)
    for i in range(10):
        print(f"{i + 1} | Risk Score: {risk_scores[i]:.3f} | Alert: {alerts[i]}")

a=1
if a==1:
    full_analysis()
plot_res()
ROC_AUC()
confusion_met()