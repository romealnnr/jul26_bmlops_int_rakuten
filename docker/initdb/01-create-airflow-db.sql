-- Airflow keeps its metadata in a database of its own, separate from the
-- application data. Postgres runs this only on first initialisation of the
-- data volume, which is also when POSTGRES_DB is honoured.
CREATE DATABASE airflow;
