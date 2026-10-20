import unittest, json
class T(unittest.TestCase):
    def test_event_payload(self):
        p={"transaction_id":1,"customer_id":100001,"amount":2500.0,"channel":"ONLINE","distance_from_home_km":12.3}
        self.assertEqual(json.loads(json.dumps(p)),p)
    def test_stream_summary(self):
        s=json.load(open("out/stream.json"))
        for k in ["events","mean_ms","p95_ms","max_ms","alerts"]: self.assertIn(k,s)
        self.assertLess(s["max_ms"],1000)
