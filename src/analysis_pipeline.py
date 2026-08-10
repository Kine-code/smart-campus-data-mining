from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, classification_report, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
from preprocessing import preprocess

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"results/figures"; MET=ROOT/"results/metrics"; MODELS=ROOT/"models"
for p in [FIG,MET,MODELS]: p.mkdir(parents=True,exist_ok=True)

def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/name,dpi=170,bbox_inches="tight"); plt.close()

def eda(df):
    daily=df.set_index("timestamp").resample("1D")["energy_kwh"].sum()
    plt.figure(figsize=(9,4)); plt.plot(daily.index,daily.values); plt.title("Tổng điện năng tiêu thụ theo ngày"); plt.xlabel("Ngày"); plt.ylabel("kWh"); savefig("energy_by_day.png")
    by_hour=df.groupby("hour")["energy_kwh"].sum()
    plt.figure(figsize=(8,4)); plt.bar(by_hour.index,by_hour.values); plt.title("Điện năng theo giờ trong ngày"); plt.xlabel("Giờ"); plt.ylabel("kWh"); savefig("energy_by_hour.png")
    by_room=df.groupby("room")["energy_kwh"].sum().sort_values()
    plt.figure(figsize=(8,5)); plt.barh(by_room.index,by_room.values); plt.title("Tổng điện năng theo phòng"); plt.xlabel("kWh"); savefig("energy_by_room.png")
    sample=df[df.room==df.room.iloc[0]].head(700)
    plt.figure(figsize=(9,4)); plt.plot(sample.timestamp,sample.temperature,label="Nhiệt độ"); plt.plot(sample.timestamp,sample.humidity,label="Độ ẩm"); plt.legend(); plt.title("Nhiệt độ và độ ẩm mẫu theo thời gian"); plt.xticks(rotation=30); savefig("temperature_humidity_timeline.png")
    cols=["temperature","humidity","light","voltage","current","power","energy_kwh","occupancy","hour"]
    corr=df[cols].corr(numeric_only=True)
    plt.figure(figsize=(8,7)); im=plt.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1); plt.colorbar(im,fraction=0.046,pad=0.04); plt.xticks(range(len(cols)),cols,rotation=45,ha="right",fontsize=8); plt.yticks(range(len(cols)),cols,fontsize=8); plt.title("Ma trận tương quan"); savefig("correlation_heatmap.png")

def clustering(df):
    agg=df.groupby("room").agg(avg_temperature=("temperature","mean"),avg_humidity=("humidity","mean"),avg_power=("power","mean"),total_energy=("energy_kwh","sum"),occupancy_rate=("occupancy","mean"),avg_light=("light","mean")).reset_index()
    X=agg.drop(columns="room")
    Xs=StandardScaler().fit_transform(X)
    scores={}; inertias={}
    for k in range(2,7):
        km=KMeans(n_clusters=k,random_state=42,n_init=20).fit(Xs)
        scores[k]=silhouette_score(Xs,km.labels_); inertias[k]=km.inertia_
    best_k=max(scores,key=scores.get)
    km=KMeans(n_clusters=best_k,random_state=42,n_init=50).fit(Xs)
    agg["cluster"]=km.labels_
    pca=PCA(n_components=2,random_state=42); pts=pca.fit_transform(Xs)
    agg["pc1"],agg["pc2"]=pts[:,0],pts[:,1]
    plt.figure(figsize=(7,5))
    for c in sorted(agg.cluster.unique()):
        d=agg[agg.cluster==c]; plt.scatter(d.pc1,d.pc2,s=80,label=f"Cluster {c}")
        for _,r in d.iterrows(): plt.text(r.pc1+0.03,r.pc2+0.03,r.room,fontsize=8)
    plt.legend(); plt.title(f"PCA trực quan hóa K-Means (K={best_k})"); plt.xlabel("PC1"); plt.ylabel("PC2"); savefig("cluster_pca.png")
    plt.figure(figsize=(6,4)); plt.plot(list(inertias),list(inertias.values()),marker="o"); plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow Method"); savefig("cluster_elbow.png")
    agg.to_csv(MET/"room_clusters.csv",index=False)
    return {"best_k":int(best_k),"silhouette":float(scores[best_k]),"scores":scores}

def time_split(df, frac_train=.7, frac_val=.15):
    df=df.sort_values("timestamp")
    times=np.array(sorted(df.timestamp.unique()))
    t1=times[int(len(times)*frac_train)]; t2=times[int(len(times)*(frac_train+frac_val))]
    return df[df.timestamp<t1],df[(df.timestamp>=t1)&(df.timestamp<t2)],df[df.timestamp>=t2]

def classification(df):
    df=df.iloc[::6].copy()
    train,val,test=time_split(df)
    features=["temperature","humidity","light","voltage","current","power_roll_mean_6","energy_lag1","hour","weekday","occupancy","peak_hour"]
    Xtr,ytr=train[features],train["target_activity_next"]
    Xv,yv=val[features],val["target_activity_next"]
    Xte,yte=test[features],test["target_activity_next"]
    models={
      "Decision Tree":DecisionTreeClassifier(max_depth=12,random_state=42,class_weight="balanced"),
      "KNN":Pipeline([("scaler",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=9))]),
      "SVM":Pipeline([("scaler",StandardScaler()),("model",LinearSVC(C=1.0,class_weight="balanced",random_state=42,dual="auto"))]),
      "Random Forest":RandomForestClassifier(n_estimators=80,max_depth=16,min_samples_leaf=2,random_state=42,n_jobs=-1,class_weight="balanced_subsample")
    }
    rows=[]; best=None; best_f=-1
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xv)
        p,r,f,_=precision_recall_fscore_support(yv,pred,average="weighted",zero_division=0)
        acc=accuracy_score(yv,pred); rows.append([name,acc,p,r,f])
        if f>best_f: best_f=f; best=name
    final=clone(models[best]); Xtv=pd.concat([Xtr,Xv]); ytv=pd.concat([ytr,yv]); final.fit(Xtv,ytv); pred=final.predict(Xte)
    p,r,f,_=precision_recall_fscore_support(yte,pred,average="weighted",zero_division=0); acc=accuracy_score(yte,pred)
    labels=sorted(yte.unique()); cm=confusion_matrix(yte,pred,labels=labels)
    plt.figure(figsize=(5,4)); im=plt.imshow(cm,cmap="Blues"); plt.colorbar(im); plt.xticks(range(len(labels)),labels,rotation=30); plt.yticks(range(len(labels)),labels); plt.xlabel("Dự đoán"); plt.ylabel("Thực tế"); plt.title(f"Confusion Matrix - {best}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,cm[i,j],ha="center",va="center")
    savefig("confusion_matrix.png")
    pd.DataFrame(rows,columns=["model","accuracy_val","precision_val","recall_val","f1_val"]).to_csv(MET/"classification_comparison.csv",index=False)
    joblib.dump(final,MODELS/"classification_best.pkl")
    return {"selected_model":best,"test_accuracy":float(acc),"test_precision":float(p),"test_recall":float(r),"test_f1":float(f)}

def regression(df):
    df=df.iloc[::4].copy()
    train,val,test=time_split(df)
    features=["temperature","humidity","light","voltage","current","power_roll_mean_6","energy_lag1","hour","weekday","occupancy","peak_hour"]
    Xtr,ytr=train[features],train.target_energy_next; Xv,yv=val[features],val.target_energy_next; Xte,yte=test[features],test.target_energy_next
    models={
      "Linear Regression":Pipeline([("scaler",StandardScaler()),("model",LinearRegression())]),
      "Decision Tree":DecisionTreeRegressor(max_depth=14,random_state=42,min_samples_leaf=3),
      "Random Forest":RandomForestRegressor(n_estimators=80,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "Gradient Boosting":GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=3,random_state=42)
    }
    rows=[]; best=None; best_rmse=1e9
    for name,m in models.items():
        m.fit(Xtr,ytr); pred=m.predict(Xv); mae=mean_absolute_error(yv,pred); rmse=mean_squared_error(yv,pred)**0.5; r2=r2_score(yv,pred); rows.append([name,mae,rmse,r2])
        if rmse<best_rmse: best_rmse=rmse; best=name
    final=clone(models[best]); final.fit(pd.concat([Xtr,Xv]),pd.concat([ytr,yv])); pred=final.predict(Xte)
    mae=mean_absolute_error(yte,pred); rmse=mean_squared_error(yte,pred)**0.5; r2=r2_score(yte,pred)
    baseline=test["energy_kwh"].to_numpy(); base_rmse=mean_squared_error(yte,baseline)**0.5
    idx=np.arange(min(500,len(yte))); plt.figure(figsize=(9,4)); plt.plot(idx,yte.iloc[:len(idx)].values,label="Actual"); plt.plot(idx,pred[:len(idx)],label="Predicted",alpha=.8); plt.legend(); plt.title(f"Actual vs Predicted - {best}"); plt.xlabel("Mẫu thời gian"); plt.ylabel("kWh / 5 phút"); savefig("actual_vs_prediction.png")
    pd.DataFrame(rows,columns=["model","mae_val","rmse_val","r2_val"]).to_csv(MET/"regression_comparison.csv",index=False)
    joblib.dump(final,MODELS/"regression_best.pkl")
    return {"selected_model":best,"test_mae":float(mae),"test_rmse":float(rmse),"test_r2":float(r2),"baseline_rmse":float(base_rmse)}

def anomaly(df):
    features=["temperature","humidity","light","current","power","energy_kwh"]
    X=StandardScaler().fit_transform(df[features])
    iso=IsolationForest(n_estimators=100,contamination=0.003,max_samples=10000,random_state=42,n_jobs=-1)
    pred=(iso.fit_predict(X)==-1).astype(int)
    y=df.injected_anomaly.astype(int).to_numpy()
    p=precision_score(y,pred,zero_division=0); r=recall_score(y,pred,zero_division=0); f=f1_score(y,pred,zero_division=0)
    out=df[["timestamp","room","power","temperature","humidity","injected_anomaly"]].copy(); out["detected_anomaly"]=pred; out[out.detected_anomaly==1].to_csv(MET/"anomaly_records.csv",index=False)
    sample=df.iloc[:3000].copy(); ps=pred[:3000]
    plt.figure(figsize=(10,4)); plt.plot(sample.timestamp,sample.power,label="Power"); m=ps==1; plt.scatter(sample.timestamp[m],sample.power[m],marker="x",s=45,label="Anomaly"); plt.legend(); plt.title("Phát hiện bất thường bằng Isolation Forest"); plt.xticks(rotation=30); savefig("anomaly_detection.png")
    joblib.dump(iso,MODELS/"isolation_forest.pkl")
    return {"precision":float(p),"recall":float(r),"f1":float(f),"detected":int(pred.sum()),"known_anomalies":int(y.sum())}

def hmm_demo(df):
    # Simple discrete HMM/Viterbi: states EMPTY, NORMAL, HIGH and observations based on occupancy/light/power buckets.
    states=["EMPTY","NORMAL","HIGH"]
    start=np.log(np.array([0.55,0.35,0.10]))
    trans=np.log(np.array([[.88,.11,.01],[.10,.80,.10],[.02,.20,.78]]))
    # obs: 0=low, 1=medium, 2=high
    emit=np.log(np.array([[.88,.11,.01],[.12,.76,.12],[.02,.18,.80]]))
    d=df[df.room=="C201"].sort_values("timestamp").head(600).copy()
    obs=np.where(d.power>1300,2,np.where((d.power>350)|(d.light>250),1,0))
    T=len(obs); dp=np.full((T,3),-1e18); back=np.zeros((T,3),dtype=int); dp[0]=start+emit[:,obs[0]]
    for t in range(1,T):
        for s in range(3):
            vals=dp[t-1]+trans[:,s]; back[t,s]=int(np.argmax(vals)); dp[t,s]=vals[back[t,s]]+emit[s,obs[t]]
    path=np.zeros(T,dtype=int); path[-1]=int(np.argmax(dp[-1]))
    for t in range(T-2,-1,-1): path[t]=back[t+1,path[t+1]]
    pred=np.array([states[i] for i in path]); acc=(pred==d.activity_state.to_numpy()).mean()
    pd.DataFrame({"timestamp":d.timestamp,"room":d.room,"observed_bucket":obs,"hmm_state":pred,"true_state":d.activity_state}).to_csv(MET/"hmm_activity_demo.csv",index=False)
    return {"room":"C201","samples":T,"state_accuracy":float(acc)}

def main():
    df,cleaning=preprocess(); eda(df)
    results={"cleaning":cleaning,"clustering":clustering(df),"classification":classification(df),"regression":regression(df),"anomaly":anomaly(df),"hmm_demo":hmm_demo(df)}
    (MET/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
