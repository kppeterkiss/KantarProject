CREATE DATABASE metabaseappdb;
CREATE USER user1 WITH PASSWORD 'example2';
GRANT ALL PRIVILEGES ON DATABASE "metabaseappdb" to user1;
\i create_tables.sql