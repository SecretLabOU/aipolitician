import sqlite3
import json

def init_database():
    """Initialize SQLite database with tables for politician data"""
    conn = sqlite3.connect('Databases/politicians.db')
    cursor = conn.cursor()

    # Create tables
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS politicians (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            party TEXT NOT NULL,
            current_position TEXT,
            bio TEXT
        );

        CREATE TABLE IF NOT EXISTS policy_positions (
            id INTEGER PRIMARY KEY,
            politician_id INTEGER,
            topic TEXT NOT NULL,
            position TEXT NOT NULL,
            date_updated TEXT NOT NULL,
            FOREIGN KEY (politician_id) REFERENCES politicians (id)
        );

        CREATE TABLE IF NOT EXISTS voting_records (
            id INTEGER PRIMARY KEY,
            politician_id INTEGER,
            bill_name TEXT NOT NULL,
            vote TEXT NOT NULL,
            date TEXT NOT NULL,
            topic TEXT NOT NULL,
            FOREIGN KEY (politician_id) REFERENCES politicians (id)
        );
    ''')

    # Sample data for testing
    sample_data = {
        'politicians': [
            (1, 'John Smith', 'Independent', 'Senator', 'Experienced senator focused on economic policy'),
            (2, 'Jane Doe', 'Progressive', 'Representative', 'Advocate for healthcare reform')
        ],
        'policy_positions': [
            (1, 1, 'economy', 'Supports free market with regulatory oversight', '2024-01-01'),
            (2, 1, 'healthcare', 'Advocates for public-private healthcare system', '2024-01-01'),
            (3, 2, 'healthcare', 'Supports universal healthcare coverage', '2024-01-01'),
            (4, 2, 'climate change', 'Promotes green energy initiatives', '2024-01-01')
        ],
        'voting_records': [
            (1, 1, 'Economic Recovery Act', 'Yes', '2023-12-15', 'economy'),
            (2, 1, 'Healthcare Reform Bill', 'No', '2023-11-20', 'healthcare'),
            (3, 2, 'Green Energy Act', 'Yes', '2023-12-01', 'climate change'),
            (4, 2, 'Healthcare Reform Bill', 'Yes', '2023-11-20', 'healthcare')
        ]
    }

    # Insert sample data
    cursor.executemany('INSERT OR REPLACE INTO politicians VALUES (?,?,?,?,?)', sample_data['politicians'])
    cursor.executemany('INSERT OR REPLACE INTO policy_positions VALUES (?,?,?,?,?)', sample_data['policy_positions'])
    cursor.executemany('INSERT OR REPLACE INTO voting_records VALUES (?,?,?,?,?,?)', sample_data['voting_records'])

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_database()
