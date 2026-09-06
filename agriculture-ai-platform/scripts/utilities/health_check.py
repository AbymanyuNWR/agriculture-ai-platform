#!/usr/bin/env python3
"""
Health check script for all services
"""

import sys
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_api():
    """Check API health"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return True, "API is healthy"
        else:
            return False, f"API returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"API connection failed: {str(e)}"

def check_database():
    """Check PostgreSQL database"""
    try:
        from sqlalchemy import create_engine
        from src.api.app.config import settings
        
        engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True, "Database is healthy"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

def check_redis():
    """Check Redis"""
    try:
        import redis
        from src.api.app.config import settings
        
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        return True, "Redis is healthy"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)}"

def check_neo4j():
    """Check Neo4j"""
    try:
        from neo4j import GraphDatabase
        from src.api.app.config import settings
        
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True, "Neo4j is healthy"
    except Exception as e:
        return False, f"Neo4j connection failed: {str(e)}"

def check_mlflow():
    """Check MLflow"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            return True, "MLflow is healthy"
        else:
            return False, f"MLflow returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"MLflow connection failed: {str(e)}"

def main():
    print("=========================================")
    print("Health Check")
    print("=========================================")
    
    checks = [
        ("API", check_api),
        ("Database", check_database),
        ("Redis", check_redis),
        ("Neo4j", check_neo4j),
        ("MLflow", check_mlflow),
    ]
    
    all_healthy = True
    
    for name, check_func in checks:
        healthy, message = check_func()
        status = "✓" if healthy else "✗"
        print(f"{status} {name}: {message}")
        if not healthy:
            all_healthy = False
    
    print("=========================================")
    
    if all_healthy:
        print("All services are healthy!")
        sys.exit(0)
    else:
        print("Some services are unhealthy!")
        sys.exit(1)

if __name__ == '__main__':
    main()
