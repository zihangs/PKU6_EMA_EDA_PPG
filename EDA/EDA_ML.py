# -*- coding: utf-8 -*-

"""
EDA 事件级分类（A方案 - 修正版）

✔ 每个事件 → 全局EDA特征
✔ 不使用滑窗 / Transformer
✔ LogReg / SVM / RF
✔ 训练集自动阈值选择
✔ 10-fold Stratified CV
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from scipy import stats
from scipy.signal import welch, butter, sosfiltfilt, find_peaks
from scipy import integrate

try:
    import antropy as ant
except:
    ant = None


# ================= PATH =================

BASE = "/vsphdatastore/suzihang/EDA_process/data/event_long_10"
RAW = os.path.join(BASE, "eda_event_raw_long.npy")
LAB = os.path.join(BASE, "eda_event_labels.npy")
META = os.path.join(BASE, "eda_event_meta_long.csv")

FS = 4.0
N_SPLITS = 10
SEED = 42
np.random.seed(SEED)


# ================= FEATURE KEYS =================

GLOBAL_FEAT_KEYS = [
    "mean","std","kurt","skew",
    "activity","mobility","complexity",
    "lf","hf","lf_hf",
    "entropy","fd",
    "scr_count"
]


def higuchi_fd(x):
    x = np.asarray(x)
    n = len(x)
    if n < 10:
        return 0
    L=[]
    for k in range(1,6):
        Lk=[]
        for m in range(k):
            idx=np.arange(m,n,k)
            if len(idx)<2: continue
            diff=np.abs(np.diff(x[idx]))
            Lk.append(diff.sum())
        if len(Lk): L.append(np.mean(Lk))
    if len(L)<2: return 0
    return float(np.polyfit(np.log(range(1,len(L)+1)),np.log(L),1)[0])


def extract_global(x):
    x=np.asarray(x,dtype=float)
    x=x[np.isfinite(x)]
    if len(x)<20:
        return np.zeros(len(GLOBAL_FEAT_KEYS))

    feats={}

    feats["mean"]=np.mean(x)
    feats["std"]=np.std(x)
    feats["kurt"]=stats.kurtosis(x)
    feats["skew"]=stats.skew(x)

    diff=np.diff(x)
    feats["activity"]=np.var(x)
    feats["mobility"]=np.sqrt(np.var(diff)/(np.var(x)+1e-6))
    feats["complexity"]=np.sqrt(np.var(np.diff(diff))/(np.var(diff)+1e-6))

    f,P=welch(x,fs=FS)
    lf=np.trapz(P[(f>=.05)&(f<.15)])
    hf=np.trapz(P[(f>=.15)&(f<.4)])
    feats["lf"]=lf
    feats["hf"]=hf
    feats["lf_hf"]=lf/(hf+1e-6)

    feats["entropy"]=ant.sample_entropy(x) if ant else 0
    feats["fd"]=higuchi_fd(x)

    peaks,_=find_peaks(x,height=np.mean(x))
    feats["scr_count"]=len(peaks)

    return np.nan_to_num([feats[k] for k in GLOBAL_FEAT_KEYS])


# ================= BUILD FEATURE MATRIX =================

def build_data():
    Xraw=np.load(RAW,allow_pickle=True)
    y=np.load(LAB).astype(int)

    feats=[]
    for i,x in enumerate(Xraw):
        feats.append(extract_global(x))
        if i%200==0: print("feature",i)

    X=np.array(feats)
    print("X:",X.shape,"label:",np.bincount(y))
    return X,y


# ================= THRESHOLD SELECTION =================

def pick_threshold(y,p,metric="bal"):
    ts=np.linspace(.05,.95,19)
    best=(.5,-1)
    for t in ts:
        pred=(p>=t).astype(int)
        if metric=="f1":
            v=f1_score(y,pred,zero_division=0)
        else:
            v=balanced_accuracy_score(y,pred)
        if v>best[1]:
            best=(t,v)
    return best


# ================= CV TRAINING =================

def run_models(X,y):

    skf=StratifiedKFold(N_SPLITS,shuffle=True,random_state=SEED)

    models={

        "LogReg":Pipeline([
            ("scaler",StandardScaler()),
            ("clf",LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear"))
        ]),

        "SVM":Pipeline([
            ("scaler",StandardScaler()),
            ("clf",SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True))
        ]),

        "RF":RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced_subsample",
            n_jobs=-1)
    }

    for name,model in models.items():

        print("\n====",name,"====")
        scores=[]
        all_y=[]
        all_pred=[]

        for fold,(tr,te) in enumerate(skf.split(X,y),1):

            m=clone(model)
            m.fit(X[tr],y[tr])

            prob_tr=m.predict_proba(X[tr])[:,1]
            prob_te=m.predict_proba(X[te])[:,1]

            t,best=pick_threshold(y[tr],prob_tr)

            pred=(prob_te>=t).astype(int)

            acc=accuracy_score(y[te],pred)
            bal=balanced_accuracy_score(y[te],pred)
            f1=f1_score(y[te],pred,zero_division=0)
            auc=roc_auc_score(y[te],prob_te)

            scores.append([acc,bal,f1,auc])
            all_y.append(y[te])
            all_pred.append(pred)

            print(f"fold{fold} t={t:.2f} ACC={acc:.3f} BAL={bal:.3f} F1={f1:.3f} AUC={auc:.3f}")

        s=np.array(scores)
        print("\nmean±std")
        print("ACC",s[:,0].mean(),s[:,0].std())
        print("BAL",s[:,1].mean(),s[:,1].std())
        print("F1 ",s[:,2].mean(),s[:,2].std())
        print("AUC",s[:,3].mean(),s[:,3].std())

        cm=confusion_matrix(np.concatenate(all_y),
                            np.concatenate(all_pred))
        print("CM\n",cm)


# ================= MAIN =================

def main():

    print("\nBuild features...")
    X,y=build_data()

    print("\nRun ML models...")
    run_models(X,y)


if __name__=="__main__":
    main()
