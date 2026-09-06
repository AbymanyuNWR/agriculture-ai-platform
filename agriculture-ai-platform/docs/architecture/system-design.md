# System Design Document

## Overview

Agriculture AI Platform adalah sistem end-to-end untuk deteksi penyakit tanaman dan smart farming.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐│
│  │  Mobile App │  │  Web App    │  │  SMS Gateway│  │  IoT Hub  ││
│  │  (Flutter)  │  │  (React)    │  │  (Twilio)   │  │  (MQTT)   ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    NGINX Load Balancer                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                  │                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │   │
│  │  │   Auth   │  │   Rate   │  │   CORS   │  │ Logging  │  │   │
│  │  │  Module  │  │ Limiter  │  │  Module  │  │  Module  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   ML SERVICES     │ │  KNOWLEDGE        │ │  EXTERNAL         │
│                   │ │  SERVICES         │ │  SERVICES         │
├───────────────────┤ ├───────────────────┤ ├───────────────────┤
│ Disease Classifier│ │ Knowledge Graph   │ │ Weather API       │
│ Pest Detector     │ │ Ontology Service  │ │ Market Prices     │
│ Crop Recommender  │ │ Reasoning Engine  │ │ Satellite Data    │
│ Yield Predictor   │ │ LLM Service       │ │ IoT Data          │
└───────────────────┘ └───────────────────┘ └───────────────────┘
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐│
│  │ PostgreSQL  │  │    Redis    │  │    Neo4j    │  │   MinIO   ││
│  │ (Main DB)   │  │  (Cache)    │  │ (Knowledge) │  │ (Objects) ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘│
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  InfluxDB   │  │Elasticsearch│  │    MLflow   │              │
│  │(Time Series)│  │  (Search)   │  │  (Tracking) │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐│
│  │ Kubernetes  │  │   Docker    │  │  Terraform  │  │  Vault    ││
│  │   (EKS)     │  │ (Container) │  │   (IaC)     │  │ (Secrets) ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Image Upload**: User uploads plant image via mobile/web
2. **Preprocessing**: Image is resized, normalized, and augmented
3. **Feature Extraction**: CNN extracts visual features
4. **Classification**: Model predicts disease with confidence
5. **Knowledge Enrichment**: Query knowledge graph for details
6. **LLM Generation**: Generate farmer-friendly explanation
7. **Response**: Return diagnosis with recommendations

### ML Pipeline

```
Data Collection → Validation → Preprocessing → Training → Evaluation → Deployment
      │                                                     │
      └─────────────────── Feedback Loop ───────────────────┘
```

### Security Architecture

- **Authentication**: JWT + OAuth2
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **Rate Limiting**: Per-user and per-endpoint
- **Input Validation**: Sanitization and type checking

### Monitoring Architecture

- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger
- **Alerting**: PagerDuty + Slack

## Scalability

### Horizontal Scaling

- API servers: Auto-scale based on CPU/memory
- ML servers: GPU-based scaling
- Database: Read replicas + connection pooling

### Vertical Scaling

- ML models: Model parallelism for large models
- Database: Upgrade instance type

### Caching Strategy

- **Redis**: API responses, session data
- **CDN**: Static assets, images
- **Model Cache**: Frequently used models in memory

## Disaster Recovery

- **Backup**: Daily automated backups
- **Replication**: Multi-AZ deployment
- **Recovery**: RTO < 1 hour, RPO < 5 minutes

## Performance Requirements

- **API Response Time**: < 200ms (p95)
- **ML Inference**: < 500ms (p95)
- **Availability**: 99.9% uptime
- **Throughput**: > 1000 requests/second
