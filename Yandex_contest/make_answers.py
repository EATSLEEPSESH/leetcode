# catboost_cats_or_nums.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
# --- switches ---
FAST = True          # быстрый режим
FOLDS = 3 if FAST else 5

NUM_ITERS   = 1500 if FAST else 5000
DEPTH       = 6    if FAST else 8
LR          = 0.05 if FAST else 0.03
OD_WAIT     = 30   if FAST else 100
BORDER_CNT  = 128  if FAST else 255

SEED = 42
FOLDS = 5
ID_CANDS = ["customer","id","ID","ClientID","client","index"]

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)

def load_data():
    p = Path(".")
    train = pd.read_csv(p/"train.csv", low_memory=False)
    test  = pd.read_csv(p/"test.csv",  low_memory=False)
    assert "target" in train.columns, "нет 'target' в train.csv"

    X = to_numeric(train.drop(columns=["target"])).astype(np.float64)
    T = to_numeric(test).astype(np.float64)

    t = train["target"]
    if np.issubdtype(t.dtype, np.number):
        y = t.astype(int).to_numpy()
    else:
        s = t.astype(str).str.strip().str.lower()
        mapping = {"1":"1","0":"0","true":"1","false":"0","yes":"1","no":"0","y":"1","n":"0"}
        y = s.map(mapping).replace({"0":0,"1":1}).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("В target один класс — обучать/оценивать нельзя.")

    # убрать полностью-пустые и константы; выровнять
    empty = set(X.columns[X.isna().all()]) | set(T.columns[T.isna().all()])
    if empty:
        X = X.drop(columns=[c for c in empty if c in X.columns])
        T = T.drop(columns=[c for c in empty if c in T.columns])
    const = X.columns[X.nunique(dropna=True) <= 1].tolist()
    if const:
        X = X.drop(columns=const)
        T = T.drop(columns=[c for c in const if c in T.columns])
    common = X.columns.intersection(T.columns)
    if len(common) == 0:
        raise ValueError("после очистки нет общих признаков между train и test")
    X = X.loc[:, common]; T = T.loc[:, common]

    id_col = next((c for c in ID_CANDS if c in test.columns), None)
    customer = (test[id_col].rename("customer") if id_col else pd.Series(np.arange(len(test)), name="customer"))

    print(f"[info] X={X.shape}, T={T.shape}, positives={int(y.sum())} ({y.mean():.4%})")
    return X, y, T, customer

def guess_categorical_cols(X: pd.DataFrame, max_unique=5000, ratio=0.5) -> list[str]:
    # Категориальные — столбцы, почти целочисленные и не слишком «сплошные»
    cats = []
    n = len(X)
    for c in X.columns:
        s = X[c].dropna()
        if s.empty:
            continue
        # «почти целые» (часто это хэши/ID)
        if np.all(np.isclose(s.values, np.round(s.values), atol=1e-12)):
            u = s.nunique()
            if (u <= max_unique) or (u <= ratio * len(s)):
                cats.append(c)
    return cats

def choose_threshold(y_true, proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, proba)
    thr_y = float(thr[int(np.argmax(tpr - fpr))])
    thr_p = float(np.quantile(proba, 1 - y_true.mean()))
    return 0.5*(thr_y + thr_p)

def run_catboost_numeric(X, y, T, class_weights):
    from catboost import CatBoostClassifier, Pool
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=6.0,
        iterations=5000,
        od_type="Iter",
        od_wait=100,
        border_count=255,
        random_seed=SEED,
        class_weights=class_weights,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1
    )
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X)); pred_test = np.zeros(len(T))
    for f, (tr, va) in enumerate(skf.split(X, y), 1):
        m = CatBoostClassifier(**params)
        m.fit(Pool(X.iloc[tr], y[tr]), eval_set=Pool(X.iloc[va], y[va]), use_best_model=True)
        oof[va] = m.predict_proba(Pool(X.iloc[va]))[:,1]
        pred_test += m.predict_proba(Pool(T))[:,1] / FOLDS
        print(f"[numeric] fold {f}/{FOLDS} done")
    auc = roc_auc_score(y, oof)
    return auc, oof, pred_test

def run_catboost_categorical(X, y, T, cat_cols, class_weights):
    from catboost import CatBoostClassifier, Pool
    # Преобразуем выбранные колонки в строки — CatBoost тогда точно трактует их как категории
    Xc = X.copy(); Tc = T.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype("Int64").astype(str)
        Tc[c] = Tc[c].astype("Int64").astype(str)
    cat_idx = [Xc.columns.get_loc(c) for c in cat_cols]
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=6.0,
        iterations=5000,
        od_type="Iter",
        od_wait=100,
        random_seed=SEED,
        class_weights=class_weights,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
        one_hot_max_size=16,   # мелкие категории — one-hot, крупные — CTR
    )
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(Xc)); pred_test = np.zeros(len(Tc))
    for f, (tr, va) in enumerate(skf.split(Xc, y), 1):
        m = CatBoostClassifier(**params)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(Xc.iloc[va], y[va], cat_features=cat_idx),
              use_best_model=True)
        oof[va] = m.predict_proba(Pool(Xc.iloc[va], cat_features=cat_idx))[:,1]
        pred_test += m.predict_proba(Pool(Tc, cat_features=cat_idx))[:,1] / FOLDS
        print(f"[categorical] fold {f}/{FOLDS} done")
    auc = roc_auc_score(y, oof)
    return auc, oof, pred_test

def main():
    try:
        from catboost import CatBoostClassifier  # noqa: F401
    except ImportError:
        raise SystemExit("CatBoost не установлен. Установите: pip install catboost")

    X, y, T, customer = load_data()

    # class weights (2% позитивов)
    n = len(y); pos = y.sum(); neg = n - pos
    class_weights = [n/(2*neg), n/(2*pos)]

    # Модель 1: «числа как числа»
    auc_num, oof_num, test_num = run_catboost_numeric(X, y, T, class_weights)
    print(f"[cv] numeric AUC = {auc_num:.6f}")

    # Модель 2: «целочисленные колонки как категории»
    cat_cols = guess_categorical_cols(X)
    if len(cat_cols) >= 1:
        print(f"[info] using {len(cat_cols)} categorical columns")
        auc_cat, oof_cat, test_cat = run_catboost_categorical(X, y, T, cat_cols, class_weights)
        print(f"[cv] categorical AUC = {auc_cat:.6f}")
    else:
        auc_cat, oof_cat, test_cat = -1.0, None, None
        print("[info] no categorical-looking columns detected; skipping categorical run")

    # Выбор/бленд
    if auc_cat > auc_num + 1e-4:
        oof, test_proba, tag = oof_cat, test_cat, "categorical"
    elif auc_num > auc_cat + 1e-4:
        oof, test_proba, tag = oof_num, test_num, "numeric"
    else:
        # усредним, если близко
        if oof_cat is not None:
            oof = 0.5*(oof_num + oof_cat)
            test_proba = 0.5*(test_num + test_cat)
            tag = "blend"
        else:
            oof, test_proba, tag = oof_num, test_num, "numeric"

    auc_final = roc_auc_score(y, oof)
    thr = choose_threshold(y, oof)
    print(f"[cv] FINAL ({tag}) AUC = {auc_final:.6f}")
    print(f"[cv] chosen threshold = {thr:.6f} ; oof positive rate = {(oof>=thr).mean():.4%}")

    labels = (test_proba >= thr).astype(int)

    pd.DataFrame({
        "customer": customer,
        "target": labels
    }).to_csv("answers.csv", index=False)

    print(f"[ok] answers.csv saved in {Path.cwd().resolve()} | shape=({len(labels)}, 2) | positives={labels.mean():.4%}")

if __name__ == "__main__":
    main()
