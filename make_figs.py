import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="DejaVu Serif"
R=json.load(open("out/results.json")); M=R["metrics"]; CM=R["cms"]
c=pd.read_csv("data/processed_transactions.csv",parse_dates=["event_time"]); hot=pd.read_csv("out/hotspots.csv")
# Fig1 thresholds
fig,ax=plt.subplots(figsize=(7,2.9),dpi=200)
for a,b,col,l in [(0,25,"#86efac","LOW\n(S < 25)"),(25,45,"#fde68a","MEDIUM\n(25 ≤ S < 45)"),(45,100,"#fca5a5","HIGH (S ≥ 45)\nalert raised")]:
    ax.barh(0,b-a,left=a,color=col,edgecolor="#1e3a8a",height=.5); ax.text((a+b)/2,0,l,ha="center",va="center",fontsize=9)
ax.set_xlim(0,100); ax.set_yticks([]); ax.set_xlabel("Composite fraud risk score S (0-100)"); ax.set_title("Risk Score Thresholds",fontsize=11)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("figs/image1.png"); plt.close()
# Fig2 architecture
fig,ax=plt.subplots(figsize=(7.2,4.2),dpi=200); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,60)
def box(x,y,w,h,t,col):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.2,rounding_size=0.8",fc=col,ec="#1e3a8a",lw=1.4)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=7.5)
def arr(x1,y1,x2,y2): ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color="#334155",lw=1.4))
box(30,52,40,6,"Synthetic card transactions\n(152,000 events, 3,000 customers)","#fde68a")
cols=[(4,"#dbeafe",["Apache Flume\n(spooldir source)","Hadoop HDFS data lake\n/raw  /cleaned  /processed","Hive warehouse\nfraud_db (partitioned)"]),
      (38,"#dcfce7",["PySpark cleaning\n+ feature engineering","Risk score +\nhotspot ranking","ML training (80/20 split)\nRF / DT / LogReg"]),
      (72,"#fecaca",["Streaming producer\n(replay)","Stream processor\n+ online inference","HIGH-risk fraud\nalerts (JSONL)"])]
for x,col,ts in cols:
    for i,t in enumerate(ts): box(x,38-i*13,24,8,t,col)
    arr(x+12,38,x+12,34) if False else None
    for i in range(2): arr(x+12,38-i*13,x+12,38-i*13-5)
arr(40,52,16,47); arr(60,52,84,47); arr(28,42,38,42)
box(28,0,44,7,"Streamlit dashboard\n(Plotly charts + Folium risk map)","#e9d5ff")
arr(16,12,36,7.5); arr(50,12,50,7.5); arr(84,12,64,7.5)
plt.savefig("figs/image2.png",bbox_inches="tight"); plt.close()
# Fig3 model comparison
names=["Random Forest","Decision Tree","Logistic Regression"]; ms=[("accuracy","Accuracy"),("f1_w","Weighted F1"),("macro_f1","Macro F1"),("fraud_f1","Fraud-class F1")]
fig,ax=plt.subplots(figsize=(7,3.6),dpi=200); w=.2
for i,(k,l) in enumerate(ms):
    v=[M[n][k] for n in names]; b=ax.bar(np.arange(3)+i*w-1.5*w,v,w,label=l)
    for r,x in zip(b,v): ax.text(r.get_x()+r.get_width()/2,x+.01,f"{x:.2f}",ha="center",fontsize=6.5)
ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_ylim(0,1.12); ax.legend(fontsize=7,ncol=4,loc="upper center"); ax.set_title("Model Comparison on Hold-out Transactions",fontsize=10)
plt.tight_layout(); plt.savefig("figs/image3.png"); plt.close()
# Fig4 confusion matrices
fig,axs=plt.subplots(1,3,figsize=(10.5,3.4),dpi=200)
for ax,n in zip(axs,names):
    m=np.array(CM[n]); ax.imshow(m,cmap="Blues",norm=matplotlib.colors.LogNorm(vmin=1,vmax=m.max()))
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{m[i,j]:,}",ha="center",va="center",fontsize=11,color="white" if m[i,j]>m.max()/50 else "black")
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Legit","Fraud"]); ax.set_yticklabels(["Legit","Fraud"]); ax.set_title(n,fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("figs/image4.png"); plt.close()
# Fig5 top5 hotspots
t5=hot.head(5); fig,ax=plt.subplots(figsize=(7,2.8),dpi=200)
lab=[f"M{m} ({ct}, {ci})" for m,ct,ci in zip(t5.merchant_id,t5.category,t5.city)][::-1]
ax.barh(lab,t5.severity[::-1],color="#dc2626")
for i,v in enumerate(t5.severity[::-1]): ax.text(v+.2,i,f"{v:.2f}",va="center",fontsize=8)
ax.set_xlabel("Merchant risk severity score"); ax.set_xlim(0,t5.severity.max()*1.15); ax.set_title("Top-5 Fraud Hotspot Merchants",fontsize=10); plt.tight_layout(); plt.savefig("figs/image5.png"); plt.close()
# Dashboard-style panels
bg="#0f172a"; fg="#e2e8f0"
def dark(fig,axs):
    fig.patch.set_facecolor(bg)
    for a in np.atleast_1d(axs).ravel():
        a.set_facecolor("#1e293b"); a.tick_params(colors=fg); a.xaxis.label.set_color(fg); a.yaxis.label.set_color(fg); a.title.set_color(fg)
        for s in a.spines.values(): s.set_color("#475569")
cities={"Mumbai":(19.07,72.88),"Delhi":(28.61,77.21),"Bengaluru":(12.97,77.59),"Chennai":(13.08,80.27),"Kolkata":(22.57,88.36),"Hyderabad":(17.38,78.48),"Pune":(18.52,73.86),"Ahmedabad":(23.02,72.57)}
# image6: risk map
fig,axs=plt.subplots(1,2,figsize=(11,4.3),dpi=170,gridspec_kw=dict(width_ratios=[1.4,1])); dark(fig,axs)
a=axs[0]; fr=c.groupby("city").agg(n=("is_fraud","size"),f=("is_fraud","mean")); fr["f"]*=100
for ct,(la,lo) in cities.items():
    a.scatter(lo,la,s=fr.loc[ct,"n"]/40,c=[fr.loc[ct,"f"]],cmap="YlOrRd",vmin=fr.f.min(),vmax=fr.f.max(),edgecolor="white",alpha=.9); a.text(lo+.4,la+.4,f"{ct}\n{fr.loc[ct,'f']:.2f}%",color=fg,fontsize=7)
a.set_ylim(11,31); a.set_xlim(70,91); a.set_xlabel("Longitude"); a.set_ylabel("Latitude"); a.set_title("Geographic Fraud Radar (bubble = volume, colour = fraud %)",fontsize=9)
a=axs[1]; k=[f"Total txns\n{len(c):,}",f"Fraud rate\n{c.is_fraud.mean()*100:.2f}%",f"HIGH-risk\n{(c.risk_level=='HIGH').sum():,}"]; a.axis("off")
for i,t in enumerate(k): a.text(.5,.8-i*.3,t,ha="center",va="center",color=fg,fontsize=13,bbox=dict(boxstyle="round",fc="#334155",ec="#64748b"),transform=a.transAxes)
plt.tight_layout(); plt.savefig("figs/image6.png"); plt.close()
# image7: historical analytics
fig,axs=plt.subplots(1,2,figsize=(11,4.3),dpi=170); dark(fig,axs)
h=c.groupby("hour_of_day").is_fraud.mean()*100; axs[0].plot(h.index,h.values,color="#f87171",marker="o"); axs[0].set_title("Fraud rate by hour of day (%)",fontsize=9); axs[0].set_xlabel("Hour"); axs[0].set_ylabel("Fraud %")
ch=c.groupby("channel").is_fraud.mean()*100; axs[1].bar(ch.index,ch.values,color=["#60a5fa","#f87171","#34d399"]); axs[1].set_title("Fraud rate by channel (%)",fontsize=9)
plt.tight_layout(); plt.savefig("figs/image7.png"); plt.close()
# image8: category/merchant matrix
fig,axs=plt.subplots(1,2,figsize=(11,4.5),dpi=170); dark(fig,axs)
cc=c.groupby("merchant_category").is_fraud.mean().sort_values()*100; axs[0].barh(cc.index,cc.values,color="#fbbf24"); axs[0].set_title("Fraud rate by merchant category (%)",fontsize=9)
pv=c.pivot_table(index="city",columns="channel",values="is_fraud",aggfunc="mean")*100; im=axs[1].imshow(pv.values,cmap="YlOrRd"); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(pv.columns); axs[1].set_yticks(range(len(pv))); axs[1].set_yticklabels(pv.index); axs[1].set_title("Fraud % : city × channel matrix",fontsize=9)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]): axs[1].text(j,i,f"{pv.values[i,j]:.2f}",ha="center",va="center",fontsize=7)
plt.tight_layout(); plt.savefig("figs/image8.png"); plt.close()
print("ok")
