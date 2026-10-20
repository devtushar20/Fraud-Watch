import unittest, os, joblib, pandas as pd
from sklearn.tree import DecisionTreeClassifier
class T(unittest.TestCase):
    def test_model_file_exists(self): self.assertTrue(os.path.exists("out/rf.joblib"))
    def test_prediction_format(self):
        m=joblib.load("out/rf.joblib"); X=pd.read_csv("data/processed_transactions.csv",nrows=5)
        X=pd.concat([X[["amount","amount_ratio","distance_from_home_km","txn_count_1h","hour_of_day","day_of_week","is_weekend","is_night","is_online","is_foreign"]],*[pd.Series(0,index=X.index,name="cat_"+k) for k in sorted(["Electronics","Fashion","Gaming","Grocery","Fuel","Jewellery","Pharmacy","Restaurant","Travel","Utilities"])]],axis=1)
        p=m.predict_proba(X); self.assertEqual(p.shape,(5,2)); self.assertTrue(((p>=0)&(p<=1)).all())
    def test_decision_tree_inference(self):
        c=pd.read_csv("data/processed_transactions.csv",nrows=3000); f=["amount","amount_ratio","distance_from_home_km","txn_count_1h"]
        d=DecisionTreeClassifier(max_depth=4,random_state=1).fit(c[f],c.is_fraud); self.assertEqual(len(d.predict(c[f].head(3))),3)
