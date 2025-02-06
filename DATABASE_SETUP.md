# Database Setup Instructions

## GPU Server Setup (Already Completed)
The PostgreSQL database has been set up on the GPU server with:
- Database name: `politician_ai`
- User: `politician_ai_user`
- Remote access enabled
- Port: 35432 (changed from default 5432)

### Change PostgreSQL Port
On the GPU server, you need to update PostgreSQL to use port 35432:

1. Edit PostgreSQL configuration:
```bash
sudo nano /etc/postgresql/14/main/postgresql.conf
```

2. Change the port line to:
```
port = 35432
```

3. Restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

4. Verify the new port:
```bash
sudo netstat -tuln | grep 35432
```

## Development Machine Setup

1. Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```

2. Update the `.env` file with the appropriate database credentials:

If running on the GPU server:
```
DATABASE_URL=postgresql://politician_ai_user:your_password_here@localhost:35432/politician_ai
```

If running on your development machine:
```
DATABASE_URL=postgresql://politician_ai_user:your_password_here@<gpu_server_ip>:35432/politician_ai
```
Replace `<gpu_server_ip>` with the actual IP address of your GPU server.

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

On GPU Server:
1. PostgreSQL is running: `sudo systemctl status postgresql`
2. Port 35432 is being used: `sudo netstat -tuln | grep 35432`
3. Database exists: `sudo -u postgres psql -p 35432 -c "\l"`
4. User has proper permissions: `sudo -u postgres psql -p 35432 -c "\du"`

On Development Machine:
1. The GPU server's IP address is correct
2. You can ping the GPU server
3. Port 35432 is accessible: `nc -zv <gpu_server_ip> 35432`
4. PostgreSQL allows remote connections (check pg_hba.conf)

### Migration Issues
If migrations fail:
1. Check the database URL in `.env`
2. Ensure the database user has proper permissions
3. Check the alembic logs for specific errors

## Data Collection
The `collect_politician_data.py` script will:
1. Initialize database tables if they don't exist
2. Collect and store basic information about Trump and Biden
3. Gather statements from news sources using News API
4. Collect additional information from Wikipedia
5. Categorize statements by political topics

Make sure to set up your API key in `.env`:
- `NEWS_API_KEY` for news statements (get from https://newsapi.org/)

The script uses two main data sources:
1. News API: For recent statements and news coverage
2. Wikipedia API: For biographical information and historical statements

## Backup and Maintenance
The database files are stored in `/Databases/politician_ai/data/postgresql` on the GPU server.
Regular backups are recommended:

```bash
# On GPU server
pg_dump -U politician_ai_user politician_ai > /Databases/politician_ai/backups/backup_$(date +%Y%m%d).sql
