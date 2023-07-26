
import sqlite3
from sqlite3 import Error
import json
import time

import seaborn as sns
import matplotlib.pyplot as plt

def get_run_ids_and_model_ids(loss_database='losses.db', metrics_database='metrics.db'):
    # Connect to the first SQLite database
    conn1 = sqlite3.connect(loss_database)
    cursor1 = conn1.cursor()

    # Connect to the second SQLite database
    conn2 = sqlite3.connect(metrics_database)
    cursor2 = conn2.cursor()

    # Query the first database for all run_ids
    cursor1.execute("""
        SELECT run_id FROM loss_table
    """)
    run_ids = cursor1.fetchall()

    # Query the second database for all run_id and model_id pairs
    cursor2.execute("""
        SELECT run_id, model_id FROM classification_report
    """)
    run_model_ids = cursor2.fetchall()

    # Close the database connections
    conn1.close()
    conn2.close()

    # Filter the run_model_ids list to only include the run_ids found in the first database
    run_model_ids = [item for item in run_model_ids if item[0] in run_ids]

    return run_model_ids

def plot_losses(run_id, database_filename='losses.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_filename)
    cursor = conn.cursor()

    # Query the database for the losses associated with the given run ID
    cursor.execute("""
        SELECT losses FROM loss_table WHERE run_id = ?
    """, (run_id,))
    
    # Fetch the record
    record = cursor.fetchone()

    if record is None:
        print(f"No record found for run_id: {run_id}")
        return
    
    # Convert the JSON string back into a list
    losses = json.loads(record[0])
    
    # Plot the list of losses
    sns.lineplot(x=list(range(len(losses))), y=losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss curve for run_id: {run_id}')
    plt.savefig(f'plots/loss_curve_{run_id}.png')

    # Close the database connection
    conn.close()

def create_connection():
    conn = None
    loss_conn = None
    try:
        conn = sqlite3.connect('metrics.db')
        loss_conn = sqlite3.connect('losses.db')
        print("Successfully Connected to SQLite")
    except Error as e:
        print(e)
    return conn, loss_conn

def close_connection(conn, loss_conn):
    if (conn):
        conn.close()
    if (loss_conn):
        loss_conn.close()
    
    print("SQLite connection is closed")

def classification_report_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classification_report (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                class TEXT NOT NULL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                support INTEGER,
                model_id TEXT NOT NULL
            )
        """)
        print("Table checked, it exists or has been successfully created.")
    except Error as e:
        print(e)

def loss_table(loss_conn):
    try:
        cursor = loss_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loss_table (
                id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                losses TEXT,
                loss_type TEXT,
                model_id TEXT NOT NULL
            )
        """)
        print("Table checked, it exists or has been successfully created.")
    except Error as e:
        print(e)

def insert_losses(loss_conn, losses, loss_type, run_id, model_id):
    loss_string = json.dumps(losses)
    try:
        cursor = loss_conn.cursor()
        cursor.execute("""
            INSERT INTO loss_table (run_id, losses, loss_type, model_id)
            VALUES (?, ?, ?, ?)
        """, (run_id, loss_string, loss_type, model_id))
    except Error as e:
        print(e)
    loss_conn.commit()

def insert_report(conn, report, run_id, model_id):
    cursor = conn.cursor()
    
    for class_name, metrics in report.items():
        if class_name in ['accuracy']:
            continue  # Skip as it has a different structure
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1-score']
        support = metrics['support']

        cursor.execute("""
            INSERT INTO classification_report (run_id, class, precision, recall, f1_score, support, model_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, class_name, precision, recall, f1_score, support, model_id))

    conn.commit()

def classification_report_table_to_df(conn):
    try:
        conn = sqlite3.connect('metrics.db')
    except Error as e:
        print(e)

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM classification_report")
    # convert to pandas dataframe
    df = pd.DataFrame(cursor.fetchall(), columns=['id', 'run_id', 'class', 'precision', 'recall', 'f1_score', 'support', 'model_id'])
    return df

def initialize_tracking():
    conn, loss_conn = create_connection()
    classification_report_table(conn)
    loss_table(loss_conn)
    return conn, loss_conn

if __name__ == '__main__':
    print(get_run_ids_and_model_ids())
    plot_losses("1690401526.2948446")