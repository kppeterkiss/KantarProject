CREATE DATABASE metabaseappdb;
CREATE USER metabase WITH PASSWORD 'metabase';
GRANT CREATE ON SCHEMA public TO metabase;
GRANT ALL PRIVILEGES ON DATABASE "metabaseappdb" to metabase;
\i create_tables.sql