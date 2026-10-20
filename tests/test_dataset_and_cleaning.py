import unittest, pandas as pd
RAW=["transaction_id","customer_id","merchant_id","event_time","amount","merchant_category","channel","city","latitude","longitude","distance_from_home_km","is_foreign","txn_count_1h","is_fraud"]
class T(unittest.TestCase):
    def test_raw_schema(self):
        self.assertEqual(list(pd.read_csv("data/raw_transactions.csv",nrows=5).columns),RAW)
    def test_cleaned_ranges(self):
        c=pd.read_csv("data/processed_transactions.csv")
        self.assertTrue((c.amount>0).all() and (c.amount<=500000).all()); self.assertEqual(c.transaction_id.duplicated().sum(),0); self.assertEqual(c.customer_id.isna().sum(),0)
