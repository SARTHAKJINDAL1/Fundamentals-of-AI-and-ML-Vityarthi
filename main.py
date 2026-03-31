import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
 
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Bidirectional, LeakyReLU, Input


# ==============================
# LOAD DATA
# ==============================
def load_data():
    print("\nLoading dataset...")
    df = pd.read_csv("Phising_Training_Dataset.csv")

    X = df.drop(['key', 'Result'], axis=1).values
    y = df['Result'].replace(-1, 0).values

    X = X.reshape(X.shape[0], X.shape[1], 1)

    print("Dataset loaded successfully!")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ==============================
# MODELS
# ==============================
def model_cnn_bigru():
    return Sequential([
        Input(shape=(30,1)),
        Conv1D(32, 3),
        LeakyReLU(),
        Bidirectional(GRU(32)),
        Dense(1, activation='sigmoid')
    ])

def model_cnn_bilstm():
    return Sequential([
        Input(shape=(30,1)),
        Conv1D(32, 3),
        LeakyReLU(),
        Bidirectional(LSTM(32)),
        Dense(1, activation='sigmoid')
    ])

def model_lstm_gru():
    return Sequential([
        Input(shape=(30,1)),
        LSTM(32, return_sequences=True),
        GRU(32),
        Dense(1, activation='sigmoid')
    ])

def model_bigru():
    return Sequential([
        Input(shape=(30,1)),
        Bidirectional(GRU(32)),
        LeakyReLU(),
        Dense(1, activation='sigmoid')
    ])


# ==============================
# PLOT ALL
# ==============================
def plot_all(results, confusion_matrices, roc_data):

    plt.figure(figsize=(18, 10))

    # Accuracy Curve
    plt.subplot(2, 3, 1)
    names = list(results.keys())
    scores = [results[n] * 100 for n in names]

    plt.plot(names, scores, marker='o')
    plt.title("Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.grid()

    for i, v in enumerate(scores):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')

    # ROC Curve
    plt.subplot(2, 3, 2)
    for name in roc_data:
        fpr, tpr, roc_auc = roc_data[name]
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1])
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()

    # Confusion Matrices
    for i, (name, cm) in enumerate(confusion_matrices.items(), start=3):
        plt.subplot(2, 3, i)
        plt.imshow(cm)
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        for x in range(len(cm)):
            for y in range(len(cm)):
                plt.text(y, x, cm[x][y], ha='center', va='center')

    plt.tight_layout()
    plt.savefig("model_results.png")
    print("\nSaved: model_results.png")
    plt.show()
# ==============================
# INTERACTIVE DETECTOR
# ==============================
def interactive_detector(best_model):

    print("\n========================================")
    print("       INTERACTIVE PHISHING DETECTOR")
    print("========================================")
    print("Enter values: 1 (Legitimate), 0 (Suspicious), -1 (Phishing)")

    while True:
        try:
            s_state    = int(input("1. SSL Final State: "))
            anchor     = int(input("2. URL Anchor Score: "))
            traffic    = int(input("3. Web Traffic Score: "))
            sub_domain = int(input("4. Having Sub-Domain: "))
            tags       = int(input("5. Links in Tags: "))
            prefix     = int(input("6. Prefix-Suffix: "))

            test_input = [0] * 30

            test_input[7]  = s_state
            test_input[13] = anchor
            test_input[25] = traffic
            test_input[6]  = sub_domain
            test_input[14] = tags
            test_input[5]  = prefix

            test_input = np.array(test_input).reshape(1, 30, 1)

            prob = best_model.predict(test_input, verbose=0)[0][0]

            print("\n********************")

            if prob > 0.7:
                print(f"RESULT: SAFE WEBSITE (Confidence: {prob:.2f})")
            elif prob > 0.4:
                print(f"RESULT: SUSPICIOUS WEBSITE (Confidence: {prob:.2f})")
            else:
                print(f"ALERT: PHISHING WEBSITE (Confidence: {prob:.2f})")

            print("********************")

        except:
            print("Invalid input! Use -1, 0, or 1")

        choice = input("\nCheck another site? (y/n): ").lower()
        if choice != 'y':
            print("System shutting down. Stay safe online!")
            break

# ==============================
# MAIN
# ==============================
def run_models():

    print("========================================")
    print("   Phishing Data Security in URL Detection System")
    print("   Course: Fundamentals of AI and ML")
    print("   Name: Sarthak Jindal")
    print("   Reg No: 25BCE10189")
    print("========================================")

    x_train, x_test, y_train, y_test = load_data()

    models = {
        "CNN+BiGRU": model_cnn_bigru(),
        "CNN+BiLSTM": model_cnn_bilstm(),
        "LSTM+GRU": model_lstm_gru(),
        "BiGRU": model_bigru()
    }

    results = {}
    confusion_matrices = {}
    roc_data = {}
    table = []

    print("\nTraining models...")

    for name, model in models.items():

        print("\n" + "="*50)
        print(f"MODEL: {name}")
        print("="*50)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)

        preds = model.predict(x_test, verbose=1)
        preds_binary = (preds > 0.5).astype(int)

        acc = np.mean(preds_binary.flatten() == y_test)
        results[name] = acc

        print(f"\nAccuracy: {acc*100:.2f}%")

        print("\nClassification Report:\n")
        print(classification_report(y_test, preds_binary))

        cm = confusion_matrix(y_test, preds_binary)
        confusion_matrices[name] = cm

        print("Confusion Matrix:")
        print(cm)

        report = classification_report(y_test, preds_binary, output_dict=True)

        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        fpr, tpr, _ = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

        table.append([name, precision, recall, f1, acc, roc_auc])

    # GRAPH
    plot_all(results, confusion_matrices, roc_data)

    # FINAL METRICS TABLE
    df = pd.DataFrame(table, columns=[
        "Model", "Precision", "Recall", "F1-Score", "Accuracy", "AUC"
    ])

    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE (ALL METRICS)")
    print("="*60)
    print(df)

    # ==============================
    # CONFUSION MATRIX COMPARISON TABLE
    # ==============================
    cm_table = []

    for name, cm in confusion_matrices.items():
        tn, fp, fn, tp = cm.ravel()
        cm_table.append([name, tn, fp, fn, tp])

    cm_df = pd.DataFrame(cm_table, columns=[
        "Model", "TN", "FP", "FN", "TP"
    ])

    print("\n" + "="*60)
    print("CONFUSION MATRIX COMPARISON TABLE")
    print("="*60)
    print(cm_df)

    # BEST MODEL
    best_row = df.iloc[df["Accuracy"].idxmax()]
    best_model_name = best_row["Model"]

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Accuracy: {best_row['Accuracy']*100:.2f}%")
    print(f"AUC: {best_row['AUC']:.2f}")

    # GET ACTUAL MODEL OBJECT
    best_model = models[best_model_name]

    # RUN INTERACTIVE MODE
    interactive_detector(best_model)

if __name__ == "__main__":
    run_models()
