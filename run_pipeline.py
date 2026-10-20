import os, numpy as np, pandas as pd, json, time, joblib
for d in ("data","out","figs"): os.makedirs(d, exist_ok=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score

rng = np.random.default_rng(42)
N = 152000
cities = {"Mumbai":(19.07,72.88),"Delhi":(28.61,77.21),"Bengaluru":(12.97,77.59),"Chennai":(13.08,80.27),
          "Kolkata":(22.57,88.36),"Hyderabad":(17.38,78.48),"Pune":(18.52,73.86),"Ahmedabad":(23.02,72.57)}
cats = ["Grocery","Fuel","Restaurant","Electronics","Travel","Fashion","Pharmacy","Jewellery","Gaming","Utilities"]
cat_amt = dict(zip(cats,[6.2,6.6,6.0,8.4,8.6,7.4,6.1,9.3,6.9,7.0]))
n_cust, n_merch = 3000, 400
cust_city = rng.choice(list(cities), n_cust)
cust_scale = rng.uniform(0.6,1.6,n_cust)
merch_cat = rng.choice(cats, n_merch)
merch_city = rng.choice(list(cities), n_merch)
merch_risk = np.ones(n_merch); bad = rng.choice(n_merch, 8, replace=False); merch_risk[bad] = rng.uniform(6,14,8)

start = pd.Timestamp("2026-01-01")
t = np.sort(rng.integers(0, 90*86400, N))
hour_w = np.array([1,1,1,1,1,2,3,5,6,6,6,6,6,6,6,6,6,7,8,8,7,5,3,2],float)
ts = start + pd.to_timedelta(t, unit="s")
# bias hours toward daytime by resampling hour
hrs = rng.choice(24, N, p=hour_w/hour_w.sum())
ts = ts.normalize() + pd.to_timedelta(hrs*3600 + rng.integers(0,3600,N), unit="s")
cust = rng.integers(0, n_cust, N); mer = rng.integers(0, n_merch, N)
channel = rng.choice(["POS","ONLINE","ATM"], N, p=[0.55,0.37,0.08])
amount = np.exp(rng.normal([cat_amt[c] for c in merch_cat[mer]], 0.8)) * cust_scale[cust]
amount = np.clip(amount, 10, 400000).round(2)
home = np.array([cities[c] for c in cust_city[cust]])
mcity = merch_city[mer]
use_home = rng.random(N) < 0.82
txn_city = np.where(use_home, cust_city[cust], mcity)
coords = np.array([cities[c] for c in txn_city])
lat = coords[:,0]+rng.normal(0,.03,N); lon = coords[:,1]+rng.normal(0,.03,N)
dist = np.sqrt(((lat-home[:,0])*111)**2 + ((lon-home[:,1])*105)**2).round(1)
foreign = (rng.random(N) < 0.025).astype(int)
df = pd.DataFrame(dict(transaction_id=np.arange(1,N+1), customer_id=cust+100000, merchant_id=mer+5000,
    event_time=ts, amount=amount, merchant_category=merch_cat[mer], channel=channel, city=txn_city,
    latitude=lat.round(4), longitude=lon.round(4), distance_from_home_km=dist, is_foreign=foreign))
df = df.sort_values("event_time").reset_index(drop=True)
# velocity (txns by same customer in previous hour) via searchsorted per customer
tv = df["event_time"].values.astype("datetime64[s]").astype(np.int64)
vel = np.zeros(len(df),int)
for c, idx in df.groupby("customer_id").indices.items():
    tt = tv[idx]; vel[idx] = np.arange(len(idx)) - np.searchsorted(tt, tt-3600, side="left")
df["txn_count_1h"] = vel
# fraud label from latent risk (independent of the rule score)
hour = df.event_time.dt.hour.values
night = ((hour>=0)&(hour<=4)).astype(float)
cm = df.groupby("customer_id")["amount"].transform("median").values
ratio = df.amount.values/cm
z = (-12.5 + 1.6*np.log1p(ratio) + 2.0*night + 1.6*(df.channel.values=="ONLINE") + 2.2*df.is_foreign.values
     + 0.008*np.minimum(df.distance_from_home_km.values,800) + 0.9*np.minimum(df.txn_count_1h.values,8)
     + 1.2*np.log(merch_risk[df.merchant_id.values-5000]) + rng.normal(0,0.35,len(df)))
df["is_fraud"] = (rng.random(len(df)) < 1/(1+np.exp(-z))).astype(int)
raw = df.copy()
# inject dirt
nn = 2200
raw.loc[rng.choice(len(raw), nn, replace=False), "amount"] = np.nan
raw.loc[rng.choice(len(raw), 900, replace=False), "customer_id"] = np.nan
raw.loc[rng.choice(len(raw), 700, replace=False), "amount"] = -rng.uniform(1,500,700).round(2)
raw.loc[rng.choice(len(raw), 400, replace=False), "amount"] = 9e6
raw.loc[rng.choice(len(raw), 600, replace=False), "latitude"] = 99.0
dups = raw.sample(1800, random_state=1); raw = pd.concat([raw, dups]).sample(frac=1, random_state=2).reset_index(drop=True)
raw.to_csv("data/raw_transactions.csv", index=False)
print("raw rows", len(raw), "fraud rate", df.is_fraud.mean())

# ---- cleaning
n0 = len(raw); c = raw.copy(); steps = {}
c = c.dropna(subset=["transaction_id","customer_id","merchant_id","amount"]); steps["null_keys_or_amount"] = n0-len(c); n1=len(c)
c = c[(c.amount>0)&(c.amount<=500000)]; steps["amount_out_of_range"] = n1-len(c); n2=len(c)
c = c[c.latitude.between(6,38)]; steps["invalid_coordinates"] = n2-len(c); n3=len(c)
c = c.drop_duplicates(subset=["transaction_id"]); steps["duplicates"] = n3-len(c)
c = c.sort_values("event_time").reset_index(drop=True); c["customer_id"]=c.customer_id.astype(int)
print("clean rows", len(c), steps)

# ---- features
c["hour_of_day"]=c.event_time.dt.hour; c["day_of_week"]=c.event_time.dt.dayofweek
c["is_weekend"]=(c.day_of_week>=5).astype(int); c["is_night"]=c.hour_of_day.between(0,4).astype(int)
c["is_online"]=(c.channel=="ONLINE").astype(int)
c["cust_median_amt"]=c.groupby("customer_id")["amount"].transform("median")
c["amount_ratio"]=(c.amount/c.cust_median_amt).round(3)
# recompute velocity on cleaned data
tv = c["event_time"].values.astype("datetime64[s]").astype(np.int64); vel=np.zeros(len(c),int)
for k, idx in c.groupby("customer_id").indices.items():
    tt=tv[idx]; vel[idx]=np.arange(len(idx))-np.searchsorted(tt,tt-3600,side="left")
c["txn_count_1h"]=vel
# rule-based risk score
def risk_score(amount_ratio, dist, vel, night, online, foreign):
    s_amt = np.minimum(amount_ratio/6.0,1)*100
    s_dist = np.minimum(dist/500.0,1)*100
    s_vel = np.minimum(vel/5.0,1)*100
    s_ctx = np.minimum((night*40+online*30+foreign*30),100)
    return np.clip(0.35*s_amt+0.25*s_dist+0.25*s_vel+0.15*s_ctx,0,100)
c["risk_score"]=risk_score(c.amount_ratio,c.distance_from_home_km,c.txn_count_1h,c.is_night,c.is_online,c.is_foreign).round(2)
c["risk_level"]=np.where(c.risk_score<25,"LOW",np.where(c.risk_score<45,"MEDIUM","HIGH"))
c.to_csv("data/processed_transactions.csv", index=False)

# ---- hotspots
g = c.groupby("merchant_id").agg(txns=("transaction_id","count"), fraud_pct=("is_fraud","mean"),
    mean_amount=("amount","mean"), avg_risk=("risk_score","mean"), category=("merchant_category","first"), city=("city",lambda s:s.mode()[0]))
g = g[g.txns>=100]
g["fraud_pct"]*=100
g["severity"] = (g.fraud_pct*0.50 + g.avg_risk*0.35 + np.minimum(g.mean_amount/20000,1)*100*0.15).round(2)
hot = g.sort_values("severity",ascending=False).head(10).reset_index()
hot.to_csv("out/hotspots.csv",index=False); print(hot.head(5))

# ---- models, chronological split
feats=["amount","amount_ratio","distance_from_home_km","txn_count_1h","hour_of_day","day_of_week","is_weekend","is_night","is_online","is_foreign"]
X=pd.concat([c[feats], pd.get_dummies(c.merchant_category,prefix="cat").astype(int)],axis=1); y=c.is_fraud
sp=int(len(c)*0.8); Xtr,Xte,ytr,yte=X.iloc[:sp],X.iloc[sp:],y.iloc[:sp],y.iloc[sp:]
models={"Random Forest":RandomForestClassifier(n_estimators=100,max_depth=15,class_weight={0:1,1:5},min_samples_leaf=3,random_state=42,n_jobs=-1),
 "Decision Tree":DecisionTreeClassifier(max_depth=12,class_weight={0:1,1:5},min_samples_leaf=10,random_state=42),
 "Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=1000))}
res={}; cms={}
for n,m in models.items():
    m.fit(Xtr,ytr); p=m.predict(Xte)
    pr,rc,f,_=precision_recall_fscore_support(yte,p,average="weighted",zero_division=0)
    pf=precision_recall_fscore_support(yte,p,labels=[1],zero_division=0)
    res[n]=dict(accuracy=accuracy_score(yte,p),precision_w=pr,recall_w=rc,f1_w=f,macro_f1=f1_score(yte,p,average="macro"),
                fraud_precision=float(pf[0][0]),fraud_recall=float(pf[1][0]),fraud_f1=float(pf[2][0]))
    cms[n]=confusion_matrix(yte,p).tolist()
# rule baseline
pb=(c.risk_level.iloc[sp:]=="HIGH").astype(int)
pf=precision_recall_fscore_support(yte,pb,labels=[1],zero_division=0)
res["Rule score (HIGH)"]=dict(accuracy=accuracy_score(yte,pb),fraud_precision=float(pf[0][0]),fraud_recall=float(pf[1][0]),fraud_f1=float(pf[2][0]))
cms["Rule score (HIGH)"]=confusion_matrix(yte,pb).tolist()
joblib.dump(models["Random Forest"],"out/rf.joblib")
imp=pd.Series(models["Random Forest"].feature_importances_,index=X.columns).sort_values(ascending=False)
meta=dict(raw_rows=n0+0,clean_rows=len(c),steps=steps,train=sp,test=len(c)-sp,fraud_rate_all=float(y.mean()),
          fraud_rate_test=float(yte.mean()),fraud_test=int(yte.sum()),level_counts_test=c.risk_level.iloc[sp:].value_counts().to_dict(),
          n_customers=int(c.customer_id.nunique()),n_merchants=int(c.merchant_id.nunique()),raw_with_dups=len(raw),
          importance=imp.head(8).round(4).to_dict(), tstart=str(c.event_time.min()), tend=str(c.event_time.max()))
json.dump(dict(metrics=res,cms=cms,meta=meta),open("out/results.json","w"),indent=1,default=float)
print(json.dumps(res,indent=1)); print(meta)

# ---- streaming replay latency
rf=models["Random Forest"]; ev=Xte.iloc[:300]; lat=[]; alerts=0
for i in range(len(ev)):
    t0=time.perf_counter(); pr=rf.predict_proba(ev.iloc[[i]])[0,1]; lat.append((time.perf_counter()-t0)*1000); alerts+= pr>=0.5
json.dump(dict(events=len(ev),mean_ms=float(np.mean(lat)),p95_ms=float(np.percentile(lat,95)),max_ms=float(np.max(lat)),alerts=int(alerts)),open("out/stream.json","w"))
print("stream",np.mean(lat),np.percentile(lat,95),alerts)
