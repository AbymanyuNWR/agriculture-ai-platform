from typing import Optional, Dict, List
from neo4j import GraphDatabase

from src.api.app.config import settings

class KnowledgeService:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    async def get_disease_info(
        self,
        disease_name: str,
        crop_type: Optional[str] = None
    ) -> Dict:
        with self.driver.session() as session:
            query = """
            MATCH (d:Disease {name: $disease_name})
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (d)-[:CAUSED_BY]->(p:Pathogen)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
            OPTIONAL MATCH (d)-[:FAVORED_BY]->(k:Kondisi)
            OPTIONAL MATCH (c:Crop {name: $crop_type})-[:SUSCEPTIBLE_TO]->(d)
            RETURN d,
                   collect(DISTINCT s.name) as symptoms,
                   collect(DISTINCT p.name) as causes,
                   collect(DISTINCT {name: t.name, dosage: t.dosage, method: t.application_method}) as treatments,
                   collect(DISTINCT {humidity: k.humidity, temperature: k.temperature}) as conditions
            """
            
            result = session.run(query, disease_name=disease_name, crop_type=crop_type)
            record = result.single()
            
            if not record:
                return self.get_default_info(disease_name)
            
            return {
                "symptoms": record["symptoms"],
                "causes": record["causes"],
                "treatments": record["treatments"],
                "environmental_factors": {
                    "conditions": record["conditions"]
                },
                "prevention": await self.get_prevention_tips(disease_name)
            }
    
    async def get_prevention_tips(self, disease_name: str) -> List[str]:
        with self.driver.session() as session:
            query = """
            MATCH (d:Disease {name: $disease_name})-[:PREVENTED_BY]->(p:Prevention)
            RETURN p.name as tip
            """
            
            result = session.run(query, disease_name=disease_name)
            return [record["tip"] for record in result]
    
    def get_default_info(self, disease_name: str) -> Dict:
        return {
            "symptoms": ["Bercak pada daun", "Perubahan warna"],
            "causes": ["Infeksi jamur"],
            "treatments": [
                {"name": "Fungisida", "dosage": "2ml/liter", "method": "Semprot"}
            ],
            "environmental_factors": {
                "conditions": [{"humidity": 85, "temperature": 28}]
            },
            "prevention": [
                "Gunakan varietas tahan",
                "Jaga kebersihan lahan",
                "Atur jarak tanam"
            ]
        }
    
    async def query_diseases_by_conditions(
        self,
        humidity: float,
        temperature: float,
        crop_type: str
    ) -> List[Dict]:
        with self.driver.session() as session:
            query = """
            MATCH (c:Crop {name: $crop_type})-[:SUSCEPTIBLE_TO]->(d:Disease)
            MATCH (d)-[:FAVORED_BY]->(k:Kondisi)
            WHERE k.humidity <= $humidity AND k.temperature >= $temperature - 5
                  AND k.temperature <= $temperature + 5
            RETURN d.name as disease, d.risk_level as risk
            ORDER BY risk DESC
            """
            
            result = session.run(
                query,
                humidity=humidity,
                temperature=temperature,
                crop_type=crop_type
            )
            
            return [dict(record) for record in result]
