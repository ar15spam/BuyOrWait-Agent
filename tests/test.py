import code.build_context as bc 

def test_normalize_futures():
    request_id = "request_01"
    
    rc = bc.build_context(request_id)
    ce = rc.normalize_future_events()
    print(ce[["event_id", "direction", "amount", "cash_amount"]])
    
test_normalize_futures()