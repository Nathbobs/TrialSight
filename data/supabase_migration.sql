-- TrialSight Supabase Migration
-- Run once in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- RLS is disabled so the anon API key can read and write data.

CREATE TABLE IF NOT EXISTS dim_phase (
    phase_id SERIAL PRIMARY KEY,
    phase_label TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_condition (
    condition_id SERIAL PRIMARY KEY,
    condition_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_sponsor (
    sponsor_id SERIAL PRIMARY KEY,
    sponsor_name TEXT UNIQUE NOT NULL,
    sponsor_type TEXT
);

CREATE TABLE IF NOT EXISTS fact_trials (
    trial_id SERIAL PRIMARY KEY,
    nct_id TEXT UNIQUE NOT NULL,
    brief_title TEXT NOT NULL,
    official_title TEXT,
    overall_status TEXT,
    phase_id INTEGER REFERENCES dim_phase(phase_id),
    condition_id INTEGER REFERENCES dim_condition(condition_id),
    sponsor_id INTEGER REFERENCES dim_sponsor(sponsor_id),
    start_date DATE,
    completion_date DATE,
    brief_summary TEXT,
    eligibility_criteria TEXT,
    is_single_sponsor BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Disable RLS so the anon/service-role API key can read and write
ALTER TABLE dim_phase DISABLE ROW LEVEL SECURITY;
ALTER TABLE dim_condition DISABLE ROW LEVEL SECURITY;
ALTER TABLE dim_sponsor DISABLE ROW LEVEL SECURITY;
ALTER TABLE fact_trials DISABLE ROW LEVEL SECURITY;

-- Grant full access to anon and authenticated roles
GRANT ALL ON dim_phase, dim_condition, dim_sponsor, fact_trials TO anon, authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
