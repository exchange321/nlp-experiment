import os

import psycopg2
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class DatabaseCursor:
    def __init__(self, schema):
        self.DB_HOST = os.getenv('DB_HOST')
        self.DB_PORT = os.getenv('DB_PORT')
        self.DB_DATABASE = os.getenv('DB_DATABASE')
        self.DB_USERNAME = os.getenv('DB_USERNAME')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.DB_SCHEMA = schema

    def __enter__(self):
        self.conn = psycopg2.connect(
            host=self.DB_HOST,
            port=self.DB_PORT,
            dbname=self.DB_DATABASE,
            user=self.DB_USERNAME,
            password=self.DB_PASSWORD,
            options=f'-c search_path={self.DB_SCHEMA}'
        )

        self.cur = self.conn.cursor()
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
