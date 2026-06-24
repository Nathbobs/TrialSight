import os
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

BRONZE_DIR = Path(__file__).parent.parent / "data" / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"

CONDITIONS = ["cancer", "diabetes", "alzheimer", "hypertension", "covid"]
TARGET_PER_CONDITION = 200


def _fetch_condition(condition: str, target: int) -> list[dict]:
    """Fetch up to `target` studies for a given condition from ClinicalTrials.gov API v2."""
    studies = []
    page_token = None
    page_size = 100

    while len(studies) < target:
        params = {
            "query.cond": condition,
            "pageSize": page_size,
            "format": "json",
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,OverallStatus,"
                "Phase,StartDate,CompletionDate,Condition,"
                "InterventionName,LeadSponsorName,"
                "BriefSummary,EligibilityCriteria"
            ),
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            response = requests.get(CT_API_BASE, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[BRONZE] API error for '{condition}': {e}. Retrying once...")
            time.sleep(2)
            try:
                response = requests.get(CT_API_BASE, params=params, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e2:
                raise RuntimeError(f"[BRONZE] API failed for '{condition}' after retry: {e2}")

        data = response.json()
        batch = data.get("studies", [])

        if not batch:
            break

        for study in batch:
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            desc_mod = proto.get("descriptionModule", {})
            design_mod = proto.get("designModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            elig_mod = proto.get("eligibilityModule", {})
            cond_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})

            # Extract intervention names
            interventions = arms_mod.get("interventions", [])
            intervention_names = "; ".join(
                i.get("name", "") for i in interventions if i.get("name")
            ) if interventions else None

            # Extract conditions list
            conditions_list = cond_mod.get("conditions", [])
            condition_str = "; ".join(conditions_list) if conditions_list else condition

            # Extract phase
            phases = design_mod.get("phases", [])
            phase_str = "; ".join(phases) if phases else None

            record = {
                "nctId": id_mod.get("nctId"),
                "briefTitle": id_mod.get("briefTitle"),
                "officialTitle": id_mod.get("officialTitle"),
                "overallStatus": status_mod.get("overallStatus"),
                "phase": phase_str,
                "startDate": status_mod.get("startDateStruct", {}).get("date"),
                "completionDate": status_mod.get("completionDateStruct", {}).get("date"),
                "condition": condition_str,
                "searchCondition": condition,
                "intervention": intervention_names,
                "sponsorName": sponsor_mod.get("leadSponsor", {}).get("name"),
                "briefSummary": desc_mod.get("briefSummary"),
                "eligibilityCriteria": elig_mod.get("eligibilityCriteria"),
            }
            studies.append(record)

        next_token = data.get("nextPageToken")
        if not next_token or len(studies) >= target:
            break

        page_token = next_token
        time.sleep(0.3)

    return studies[:target]


def run_bronze() -> pd.DataFrame:
    """Fetch all conditions and save raw data to Bronze Parquet."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[BRONZE] Starting extraction at {timestamp}")

    all_records = []
    for condition in CONDITIONS:
        print(f"[BRONZE] Fetching '{condition}'...")
        records = _fetch_condition(condition, TARGET_PER_CONDITION)
        print(f"[BRONZE] '{condition}': {len(records)} records fetched")
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("[BRONZE] No records fetched — aborting pipeline")

    df = pd.DataFrame(all_records)
    # Drop records missing nctId immediately (Bronze rule)
    before = len(df)
    df = df.dropna(subset=["nctId"])
    dropped = before - len(df)
    if dropped:
        print(f"[BRONZE] Dropped {dropped} records with missing nctId")

    # Deduplicate by nctId (same trial may appear under multiple conditions)
    before = len(df)
    df = df.drop_duplicates(subset=["nctId"])
    deduped = before - len(df)
    if deduped:
        print(f"[BRONZE] Deduplicated {deduped} duplicate nctIds")

    output_path = BRONZE_DIR / f"trials_raw_{timestamp}.parquet"
    df.to_parquet(output_path, index=False)

    print(f"[BRONZE] Input: {len(all_records)} → Output: {len(df)} records")
    print(f"[BRONZE] Saved to {output_path}")
    return df


if __name__ == "__main__":
    df = run_bronze()
    print(df.head())
