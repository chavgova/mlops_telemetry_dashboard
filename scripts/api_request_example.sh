# Normal Call
curl \
-H "Content-Type: application/json" \
-d '{"amount":123.00, "category":"food", "payment_type": "1"}' \
 -X POST http://127.0.0.1:5001/predict

 # Negative Call
curl \
-H "Content-Type: application/json" \
-d '{"amount":-222.00, "category":"food", "payment_type": "1"}' \
 -X POST http://127.0.0.1:5001/predict

 # Unknown category
curl \
-H "Content-Type: application/json" \
-d '{"amount":222.00, "category":"thiscatdoesntexist", "payment_type": "1"}' \
 -X POST http://127.0.0.1:5001/predict
