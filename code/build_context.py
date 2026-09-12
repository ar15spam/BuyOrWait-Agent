import pandas as pd
from pathlib import Path
import datetime
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

profiles = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
events = pd.read_csv(DATASET_DIR / "financial_events.csv")
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")

samples["request_date"] = pd.to_datetime(samples["request_date"])
events["settlement_date"] = pd.to_datetime(events["settlement_date"])



@dataclass
class RequestContext:
    request: pd.Series
    profile: pd.Series
    future_events: pd.DataFrame
    
    def normalize_future_events(self):
        clean_events = self.future_events.copy()
        
        exclude = ["cancelled", "failed", "unrealized"]
        
        valid = (
            ~clean_events["status"].isin(exclude)
            &clean_events["amount"].notna()
            & ~(
                (clean_events["status"] == "pending")
                & (clean_events["direction"] == "credit")
            )
        )
        
        clean_events = clean_events[valid].copy()
        
        clean_events["cash_amount"] = clean_events["amount"].where(
            clean_events["direction"] == "credit",
            -clean_events["amount"],
        )
        
        return clean_events
        
    
def build_context(request_id: str) -> RequestContext:
    request = get_request(request_id)
    user_id = get_user_id(request)
    user_profile = get_user_profile(user_id)
    user_future_events = get_future_events(get_user_events(user_id), request["request_date"])
    
    rc = RequestContext(request, user_profile, user_future_events)
    return rc
    
#['settled', 'cancelled', 'pending', 'scheduled', 'failed', 'unrealized'] 
def getPossibleStatuses(): 
    return events["status"].unique()
    
def get_request(request_id: str) -> pd.Series:
    request = samples[samples["request_id"] == request_id].iloc[0]
    if request.empty:
        raise ValueError("couldn't find a profile for request_id :{request_id}")
    return request
    
    
def get_user_id(request: pd.Series) -> str:
    return request["user_id"]

def get_request_date(request_id) -> pd.Timestamp:
    return get_request(request_id)["request_date"]
        
        
def get_user_profile(user_id: str) -> pd.Series: 
    user_profile = profiles[profiles["user_id"] == user_id].iloc[0]
    if user_profile.empty:
        raise ValueError("couldn't find a profile for this userid")
    else: 
        return user_profile
        
def get_user_events(user_id: str) -> pd.DataFrame:
    user_events = events[events["user_id"] == user_id]
    if user_events.empty: 
        raise ValueError("couldn't get any user events")
    else:
        return user_events
    
    
def get_future_events(user_events: pd.DataFrame, request_date: pd.Timestamp) -> pd.DataFrame:
    future_events = user_events[
        user_events["settlement_date"] >= request_date
    ].copy()

    return future_events.sort_values("settlement_date")


def normalize_future_events(future_events: pd.DataFrame):
    clean_events = future_events.copy()
    
    exclude = ["cancelled", "failed", "unrealized"]
    
    valid = (
        ~clean_events["status"].isin(exclude)
        &clean_events["amount"].notna()
        & ~(
            (clean_events["status"] == "pending")
            & (clean_events["direction"] == "credit")
        )
    )
    
    clean_events = clean_events[valid].copy()
    
    return clean_events

