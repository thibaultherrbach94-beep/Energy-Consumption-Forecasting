import os
import psycopg
import pandas as pd


def get_connection():
    """
    Ouvre une connexion PostgreSQL.

    Le mot de passe est lu depuis la variable
    d'environnement POSTGRES_PASSWORD.
    """

    password = os.getenv(
        "POSTGRES_PASSWORD"
    )

    if password is None:
        raise RuntimeError(
            "Variable POSTGRES_PASSWORD absente."
        )

    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname="energy_forecasting",
        user="postgres",
        password=password,
        connect_timeout=5
    )


def load_ml_dataset(
    start,
    end
):
    """
    Charge le dataset ML entre start et end.
    """

    query = """
        SELECT *

        FROM energy.ml_dataset

        WHERE
            timestamp_utc >= %s

        AND
            timestamp_utc < %s

        ORDER BY timestamp_utc;
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                (start, end)
            )

            rows = cur.fetchall()

            columns = [
                desc.name
                for desc
                in cur.description
            ]

    return pd.DataFrame(
        rows,
        columns=columns
    )