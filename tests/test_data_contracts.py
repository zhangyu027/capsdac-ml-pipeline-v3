import pandas as pd
from src.capsdac_ml.data_contracts import validate_enrollment_snapshot

def test_validate_enrollment_snapshot_passes():
    df = pd.DataFrame({
        "SnapshotMonth": ["2026-01"],
        "VendorNumber": [123],
        "VendorName": ["Vendor"],
        "PreschoolCDSCode": ["001"],
        "SiteName": ["Site"],
        "County": ["California"],
        "EnrollmentCount": [10],
        "FundingType": ["CSPP"],
        "DataSource": ["sample"],
    })
    validate_enrollment_snapshot(df)
