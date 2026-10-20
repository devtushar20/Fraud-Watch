import unittest, pandas as pd, numpy as np
def risk_score(r,d,v,n,o,f):
    return float(np.clip(0.35*min(r/6,1)*100+0.25*min(d/500,1)*100+0.25*min(v/5,1)*100+0.15*min(n*40+o*30+f*30,100),0,100))
class T(unittest.TestCase):
    def test_score_bounds_and_high(self):
        c=pd.read_csv("data/processed_transactions.csv"); self.assertTrue(c.risk_score.between(0,100).all())
        self.assertGreaterEqual(risk_score(8,700,6,1,1,1),45); self.assertLess(risk_score(1,2,0,0,0,0),25)
    def test_derived_columns(self):
        c=pd.read_csv("data/processed_transactions.csv",nrows=100)
        for k in ["hour_of_day","day_of_week","is_weekend","is_night","amount_ratio","txn_count_1h","risk_score","risk_level"]: self.assertIn(k,c.columns)
