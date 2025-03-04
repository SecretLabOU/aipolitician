"""
Database configuration for the Political RAG system.
This module provides configuration for connecting to databases stored outside the repository.
"""
import os
from pathlib import Path

# Base directory for all databases (outside git repo)
DATABASE_ROOT = Path('/home/natalie/Databases/political_rag')

# Ensure the database directories exist
os.makedirs(DATABASE_ROOT, exist_ok=True)

# Individual database file paths
BIOGRAPHY_DB = DATABASE_ROOT / 'biography.db'
POLICY_DB = DATABASE_ROOT / 'policy.db'
VOTING_RECORD_DB = DATABASE_ROOT / 'voting_record.db'
PUBLIC_STATEMENTS_DB = DATABASE_ROOT / 'public_statements.db'
FACT_CHECK_DB = DATABASE_ROOT / 'fact_check.db'
TIMELINE_DB = DATABASE_ROOT / 'timeline.db'
LEGISLATIVE_DB = DATABASE_ROOT / 'legislative.db'
CAMPAIGN_PROMISES_DB = DATABASE_ROOT / 'campaign_promises.db'
EXECUTIVE_ACTIONS_DB = DATABASE_ROOT / 'executive_actions.db'
MEDIA_COVERAGE_DB = DATABASE_ROOT / 'media_coverage.db'
PUBLIC_OPINION_DB = DATABASE_ROOT / 'public_opinion.db'
CONTROVERSIES_DB = DATABASE_ROOT / 'controversies.db'
POLICY_COMPARISON_DB = DATABASE_ROOT / 'policy_comparison.db'
JUDICIAL_APPOINTMENTS_DB = DATABASE_ROOT / 'judicial_appointments.db'
FOREIGN_POLICY_DB = DATABASE_ROOT / 'foreign_policy.db'
ECONOMIC_METRICS_DB = DATABASE_ROOT / 'economic_metrics.db'
CHARITY_DB = DATABASE_ROOT / 'charity.db'

# Map of database names to file paths
DATABASE_PATHS = {
    'biography': BIOGRAPHY_DB,
    'policy': POLICY_DB,
    'voting_record': VOTING_RECORD_DB,
    'public_statements': PUBLIC_STATEMENTS_DB,
    'fact_check': FACT_CHECK_DB,
    'timeline': TIMELINE_DB,
    'legislative': LEGISLATIVE_DB,
    'campaign_promises': CAMPAIGN_PROMISES_DB,
    'executive_actions': EXECUTIVE_ACTIONS_DB,
    'media_coverage': MEDIA_COVERAGE_DB,
    'public_opinion': PUBLIC_OPINION_DB,
    'controversies': CONTROVERSIES_DB,
    'policy_comparison': POLICY_COMPARISON_DB,
    'judicial_appointments': JUDICIAL_APPOINTMENTS_DB,
    'foreign_policy': FOREIGN_POLICY_DB,
    'economic_metrics': ECONOMIC_METRICS_DB,
    'charity': CHARITY_DB,
}