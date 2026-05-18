import json
import os
import requests

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fitness_assistant")
POSTGRES_USER = os.getenv("POSTGRES_USER", "swetha")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fitnesspass")


def create_datasource():
    datasource = {
        "name": "PostgreSQL",
        "type": "postgres",
        "url": f"{POSTGRES_HOST}:5432",
        "access": "proxy",
        "user": POSTGRES_USER,
        "secureJsonData": {"password": POSTGRES_PASSWORD},
        "jsonData": {
            "database": POSTGRES_DB,
            "sslmode": "disable",
            "maxOpenConns": 100,
            "maxIdleConns": 100,
            "connMaxLifetime": 14400,
            "postgresVersion": 1300,
            "timescaledb": False,
        },
        "isDefault": True,
    }

    resp = requests.post(
        f"{GRAFANA_URL}/api/datasources",
        json=datasource,
        auth=(GRAFANA_USER, GRAFANA_PASSWORD),
    )

    if resp.status_code == 409:
        print("Datasource already exists — skipping.")
    elif resp.status_code in (200, 201):
        print("Datasource created successfully.")
    else:
        print(f"Failed to create datasource: {resp.status_code} {resp.text}")

    return resp


def import_dashboard():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "dashboard.json")

    with open(dashboard_path) as f:
        dashboard = json.load(f)

    payload = {
        "dashboard": dashboard,
        "overwrite": True,
        "inputs": [
            {
                "name": "DS_POSTGRES",
                "type": "datasource",
                "pluginId": "postgres",
                "value": "PostgreSQL",
            }
        ],
    }

    resp = requests.post(
        f"{GRAFANA_URL}/api/dashboards/import",
        json=payload,
        auth=(GRAFANA_USER, GRAFANA_PASSWORD),
    )

    if resp.status_code in (200, 201):
        print("Dashboard imported successfully.")
        print(f"  → Open: {GRAFANA_URL}/d/fitness-assistant")
    else:
        print(f"Failed to import dashboard: {resp.status_code} {resp.text}")

    return resp


if __name__ == "__main__":
    print("Setting up Grafana...")
    print(f"  Grafana URL: {GRAFANA_URL}")
    print(f"  Postgres:    {POSTGRES_HOST}/{POSTGRES_DB}")

    create_datasource()
    import_dashboard()

    print("\nDone! Open http://localhost:3000 (admin/admin)")
