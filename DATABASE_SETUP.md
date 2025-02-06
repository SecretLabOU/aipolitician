# Database Setup Instructions

## GPU Server Setup (Already Completed)
The PostgreSQL database has been set up on the GPU server with:
- Database name: `politician_ai`
- User: `politician_ai_user`
- Remote access enabled

## Development Machine Setup

1. Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```

2. Update the `.env` file with your GPU server's IP and database credentials:
```
DATABASE_URL=postgresql://politician_ai_user:your_password_here@your_gpu_ip:5432/politician_ai
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Run database migrations:
```bash
# Generate initial migration
alembic revision --autogenerate -m "Initial migration"

# Apply migration
alembic upgrade head
```

5. Run the data collection script:
```bash
python scripts/collect_politician_data.py
```

## Troubleshooting

### Connection Issues
If you can't connect to the database, verify:
1. The GPU server's IP address is correct
2. PostgreSQL is running on the GPU server
3. Your network allows connections to port 5432
4. The database credentials are correct

### Migration Issues
If migrations fail:
1. Check the database URL in `.env`
2. Ensure the database user has proper permissions
3. Check the alembic logs for specific errors

## Data Collection
The `collect_politician_data.py` script will:
1. Initialize database tables if they don't exist
2. Collect and store basic information about Trump and Biden
3. Gather statements from news sources
4. Store voting records and policy positions

Make sure to set up your API keys in `.env`:
- `PROPUBLICA_API_KEY` for voting records
- `NEWS_API_KEY` for news statements

## Backup and Maintenance
The database files are stored in `/Databases/politician_ai/data/postgresql` on the GPU server.
Regular backups are recommended:

```bash
# On GPU server
pg_dump -U politician_ai_user politician_ai > /Databases/politician_ai/backups/backup_$(date +%Y%m%d).sql
