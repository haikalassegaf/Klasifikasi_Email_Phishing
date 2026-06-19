# model_worker.py
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# ==========================================
# TEXT PREPROCESSING UTILITY
# ==========================================
def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = [col.lower().strip() for col in df.columns]

    if not all(col in df.columns for col in ["subject", "body", "label"]):
        raise ValueError("Dataset wajib mengandung kolom: subject, body, dan label.")

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df.dropna(subset=["label"]).copy()

    df["text"] = df["subject"].astype(str) + " " + df["body"].astype(str)
    
    # Preprocessing teks dasar
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"] != ""].copy()
    
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])
    
    return df, label_encoder

def get_vectorizer(method_name):
    """Mengembalikan konfigurasi Vectorizer berdasarkan metode"""
    if "TF-IDF" in method_name:
        return TfidfVectorizer(max_features=4000, stop_words='english')
    elif "CountVectorizer" in method_name:
        return CountVectorizer(max_features=4000, stop_words='english')
    return None

def get_classifier(method_name):
    """Mengembalikan objek Model Classifier berdasarkan nama metode"""
    if "SVM" in method_name:
        return LinearSVC(random_state=42, max_iter=2000)
    elif "Naive Bayes" in method_name or "MultinomialNB" in method_name:
        return MultinomialNB(alpha=1.0)
    elif "Gaussian NB" in method_name:
        return GaussianNB()
    raise ValueError("Kombinasi model klasifikasi tidak dikenali.")

def build_avg_word2vec(tokens, w2v_model):
    """Helper untuk menghitung rata-rata vektor Word2Vec dari sebuah dokumen"""
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

# ==========================================
# MAIN EXPERIMENT PROCESSING PIPELINE
# ==========================================
def run_experiment(dataset_path, method):
    df, label_encoder = load_and_clean_data(dataset_path)
    
    X = df["clean_text"].values
    y = df["label_encoded"].values

    # Pisahkan Train-Test Set awal (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies, cv_precisions, cv_recalls, cv_f1_scores = [], [], [], []

    # --------------------------------------------------
    # SKENARIO A: MENGGUNAKAN TF-IDF / COUNTVECTORIZER
    # --------------------------------------------------
    if "Word2Vec" not in method:
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            vectorizer = get_vectorizer(method)
            X_tr_vec = vectorizer.fit_transform(X_tr)
            X_val_vec = vectorizer.transform(X_val)

            model = get_classifier(method)
            model.fit(X_tr_vec, y_tr)
            preds = model.predict(X_val_vec)

            cv_accuracies.append(accuracy_score(y_val, preds))
            cv_precisions.append(precision_score(y_val, preds, average='macro', zero_division=0))
            cv_recalls.append(recall_score(y_val, preds, average='macro', zero_division=0))
            cv_f1_scores.append(f1_score(y_val, preds, average='macro', zero_division=0))

        # Model Final
        final_vectorizer = get_vectorizer(method)
        X_train_final = final_vectorizer.fit_transform(X_train)
        X_test_final = final_vectorizer.transform(X_test)
        
        final_model = get_classifier(method)
        final_model.fit(X_train_final, y_train)
        y_test_pred = final_model.predict(X_test_final)

    # --------------------------------------------------
    # SKENARIO B: MENGGUNAKAN EMBEDDING WORD2VEC
    # --------------------------------------------------
    else:
        # Tokenisasi seluruh dokumen teks untuk kebutuhan Word2Vec
        train_tokens = [simple_preprocess(text) for text in X_train]
        test_tokens = [simple_preprocess(text) for text in X_test]

        for train_idx, val_idx in skf.split(X_train, y_train):
            # Mengambil subset token berdasarkan indeks fold
            tokens_tr = [train_tokens[i] for i in train_idx]
            tokens_val = [train_tokens[i] for i in val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Train Model Word2Vec secara lokal per lipatan data
            w2v = Word2Vec(sentences=tokens_tr, vector_size=100, window=5, min_count=1, workers=4)
            
            X_tr_vec = np.array([build_avg_word2vec(t, w2v) for t in tokens_tr])
            X_val_vec = np.array([build_avg_word2vec(t, w2v) for t in tokens_val])

            model = get_classifier(method)
            
            # Jika menggunakan GaussianNB pada Word2Vec, konversi matriks ke bentuk Dense array numpy
            if isinstance(model, GaussianNB) and hasattr(X_tr_vec, "toarray"):
                X_tr_vec = X_tr_vec.toarray()
                X_val_vec = X_val_vec.toarray()
                
            model.fit(X_tr_vec, y_tr)
            preds = model.predict(X_val_vec)

            cv_accuracies.append(accuracy_score(y_val, preds))
            cv_precisions.append(precision_score(y_val, preds, average='macro', zero_division=0))
            cv_recalls.append(recall_score(y_val, preds, average='macro', zero_division=0))
            cv_f1_scores.append(f1_score(y_val, preds, average='macro', zero_division=0))

        # Train Model Final Word2Vec pada seluruh Training Set
        final_w2v = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)
        X_train_final = np.array([build_avg_word2vec(t, final_w2v) for t in train_tokens])
        X_test_final = np.array([build_avg_word2vec(t, final_w2v) for t in test_tokens])

        final_model = get_classifier(method)
        final_model.fit(X_train_final, y_train)
        y_test_pred = final_model.predict(X_test_final)

    # Output kompilasi data metrik evaluasi
    return {
        "classes": [str(cls) for cls in label_encoder.classes_],
        "cv_metrics": {
            "accuracy": np.mean(cv_accuracies),
            "precision": np.mean(cv_precisions),
            "recall": np.mean(cv_recalls),
            "f1": np.mean(cv_f1_scores)
        },
        "test_metrics": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_test_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        }
    }