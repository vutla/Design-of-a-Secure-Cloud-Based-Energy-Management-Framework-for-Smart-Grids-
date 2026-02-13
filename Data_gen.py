import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from load_save import save


def Data_gen():
    data_path = "Dataset/cicddos2019_dataset.csv"
    df = pd.read_csv(data_path)

    print("Dataset Shape:", df.shape)

    X = df.drop(columns=["Label", "Class"])
    y = df["Class"]

    label_encoders = {}

    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target label
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    # MEAN IMPUTATION

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    #  NORMALIZATION (MIN-MAX)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,y,test_size=0.2,random_state=42,stratify=y)

    save("X_train", X_train)
    save("X_test", X_test)
    save("y_train", y_train)
    save("y_test", y_test)