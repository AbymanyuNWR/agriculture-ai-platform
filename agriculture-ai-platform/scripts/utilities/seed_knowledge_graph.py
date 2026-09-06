#!/usr/bin/env python3
"""
Seed knowledge graph with agricultural knowledge
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neo4j import GraphDatabase
from src.api.app.config import settings

# Disease knowledge base
DISEASES = [
    {
        "name": "Blast",
        "scientific_name": "Magnaporthe oryzae",
        "crops": ["Padi"],
        "symptoms": ["Bercak oval pada daun", "Warna abu-abu kecoklatan", "Pinggiran bercak tidak beraturan"],
        "causes": ["Jamur Magnaporthe oryzae"],
        "conditions": {"humidity": 85, "temperature": 28},
        "treatments": [
            {"name": "Tricyclazole", "dosage": "2ml/liter", "method": "Semprot"}
        ],
        "prevention": ["Gunakan varietas tahan", "Atur jarak tanam", "Jaga kebersihan lahan"]
    },
    {
        "name": "Brown Spot",
        "scientific_name": "Bipolaris oryzae",
        "crops": ["Padi"],
        "symptoms": ["Bercak coklat bulat", "Ukuran 1-5mm", "Bercak mengering"],
        "causes": ["Jamur Bipolaris oryzae"],
        "conditions": {"humidity": 80, "temperature": 25},
        "treatments": [
            {"name": "Mankozeb", "dosage": "3g/liter", "method": "Semprot"}
        ],
        "prevention": ["Gunakan benih sehat", "Pupuk seimbang", "Pengolahan tanah baik"]
    },
    {
        "name": "Leaf Blight",
        "scientific_name": "Xanthomonas oryzae",
        "crops": ["Padi"],
        "symptoms": ["Hujung daun menguning", "Bercak memanjang", "Daun layu"],
        "causes": ["Bakteri Xanthomonas oryzae"],
        "conditions": {"humidity": 90, "temperature": 30},
        "treatments": [
            {"name": "Kasugamycin", "dosage": "3ml/liter", "method": "Semprot"}
        ],
        "prevention": ["Gunakan varietas tahan", "Hindari genangan air"]
    },
    {
        "name": "Wereng",
        "scientific_name": "Nilaparvata lugens",
        "crops": ["Padi"],
        "symptoms": ["Daun menguning", "Pertumbuhan terhambat", "Tanaman mati"],
        "causes": ["Hama wereng batang coklat"],
        "conditions": {"humidity": 70, "temperature": 28},
        "treatments": [
            {"name": "Imidakloprid", "dosage": "1ml/liter", "method": "Semprot"}
        ],
        "prevention": ["Gunakan varietas tahan", "Pengendalian hayati"]
    }
]

CROPS = [
    {"name": "Padi", "species": "Oryza sativa", "conditions": {"humidity": 70, "temperature": 28}},
    {"name": "Jagung", "species": "Zea mays", "conditions": {"humidity": 60, "temperature": 25}},
    {"name": "Kedelai", "species": "Glycine max", "conditions": {"humidity": 65, "temperature": 27}}
]

def seed_knowledge_graph():
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create crops
        for crop in CROPS:
            session.run("""
                CREATE (c:Crop {
                    name: $name,
                    species: $species,
                    humidity: $humidity,
                    temperature: $temperature
                })
            """, **crop, **crop["conditions"])
        
        # Create diseases
        for disease in DISEASES:
            # Create disease node
            session.run("""
                CREATE (d:Disease {
                    name: $name,
                    scientific_name: $scientific_name,
                    humidity: $humidity,
                    temperature: $temperature
                })
            """, disease, **disease["conditions"])
            
            # Create symptom nodes
            for symptom in disease["symptoms"]:
                session.run("""
                    MATCH (d:Disease {name: $disease_name})
                    MERGE (s:Symptom {name: $symptom})
                    MERGE (d)-[:HAS_SYMPTOM]->(s)
                """, disease_name=disease["name"], symptom=symptom)
            
            # Create cause nodes
            for cause in disease["causes"]:
                session.run("""
                    MATCH (d:Disease {name: $disease_name})
                    MERGE (p:Pathogen {name: $cause})
                    MERGE (d)-[:CAUSED_BY]->(p)
                """, disease_name=disease["name"], cause=cause)
            
            # Create treatment nodes
            for treatment in disease["treatments"]:
                session.run("""
                    MATCH (d:Disease {name: $disease_name})
                    MERGE (t:Treatment {
                        name: $name,
                        dosage: $dosage,
                        method: $method
                    })
                    MERGE (d)-[:TREATED_BY]->(t)
                """, disease_name=disease["name"], **treatment)
            
            # Create prevention nodes
            for prevention in disease["prevention"]:
                session.run("""
                    MATCH (d:Disease {name: $disease_name})
                    MERGE (p:Prevention {name: $prevention})
                    MERGE (d)-[:PREVENTED_BY]->(p)
                """, disease_name=disease["name"], prevention=prevention)
            
            # Create relationships with crops
            for crop_name in disease["crops"]:
                session.run("""
                    MATCH (c:Crop {name: $crop_name})
                    MATCH (d:Disease {name: $disease_name})
                    MERGE (c)-[:SUSCEPTIBLE_TO]->(d)
                """, crop_name=crop_name, disease_name=disease["name"])
        
        # Create environmental condition nodes
        session.run("""
            CREATE (k:Kondisi {name: 'Humidity High', humidity: 85, temperature: 25})
        """)
        
        print("Knowledge graph seeded successfully!")
        
        # Print statistics
        result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
        print("\nNode counts:")
        for record in result:
            print(f"  {record['label']}: {record['count']}")
        
        result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
        print("\nRelationship counts:")
        for record in result:
            print(f"  {record['type']}: {record['count']}")
    
    driver.close()

if __name__ == '__main__':
    seed_knowledge_graph()
