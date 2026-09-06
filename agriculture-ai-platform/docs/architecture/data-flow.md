# Data Flow Document

## Overview

Dokumentasi aliran data dalam Agriculture AI Platform.

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  Images  │  │  Sensors │  │  Weather │  │   User   │     │
│  │  (Upload)│  │  (IoT)   │  │   API    │  │  Input   │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API Gateway                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  Rate    │  │  Auth    │  │  Validat. │            │   │
│  │  │ Limiter  │  │  Check   │  │  Input   │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Message Queue (Redis/RabbitMQ)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Image Processing                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Resize   │  │ Normalize│  │ Augment  │            │   │
│  │  │ 224x224  │  │ [0,1]    │  │ (Train)  │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Sensor Data Processing                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Interpol.│  │ Scaling  │  │ Feature  │            │   │
│  │  │ Missing  │  │ [0,1]    │  │ Extract  │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML INFERENCE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Feature Extraction                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │   CNN    │  │   MLP    │  │   LSTM   │            │   │
│  │  │ (Image)  │  │ (Sensor) │  │ (Time)   │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Multi-Modal Fusion                          │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │         Attention / Concatenation                │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Classification Head                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │    Softmax → Class Probabilities                │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  KNOWLEDGE ENRICHMENT                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Knowledge Graph Query                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Disease  │  │ Treatment│  │ Environ. │            │   │
│  │  │  Info    │  │  Options │  │ Factors  │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LLM Response Generation                     │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │   Prompt → Generate → Validate → Format         │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA STORAGE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  PostgreSQL  │  │    Redis     │  │    Neo4j     │        │
│  │  (Metadata)  │  │   (Cache)    │  │ (Knowledge)  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │    MinIO     │  │  InfluxDB    │  │    MLflow    │        │
│  │  (Images)    │  │ (TimeSeries) │  │  (Tracking)  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  API Response│  │  Dashboard   │  │   Alerts     │        │
│  │   (JSON)     │  │   (Chart)    │  │  (Push/Email)│        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Data Flow

### 1. Image Upload Flow

```
User Upload Image
       │
       ▼
┌──────────────────┐
│ Validate Image   │
│ - File type      │
│ - File size      │
│ - Dimensions     │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Store Original   │
│ - MinIO/S3       │
│ - Generate URL   │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Preprocess Image │
│ - Resize 224x224 │
│ - Normalize      │
│ - To Tensor      │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Run Inference    │
│ - CNN Model      │
│ - Get Predictions│
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Enrich Results   │
│ - Knowledge Graph│
│ - Weather Data   │
│ - Soil Data      │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Generate Response│
│ - Diagnosis      │
│ - Recommendations│
│ - Explanation    │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Store Results    │
│ - PostgreSQL     │
│ - Update History │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Return to User   │
│ - JSON Response  │
│ - Push Notification│
└──────────────────┘
```

### 2. Sensor Data Flow

```
IoT Sensor Data
       │
       ▼
┌──────────────────┐
│ MQTT Broker      │
│ - Receive Data   │
│ - Validate       │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Process Data     │
│ - Parse JSON     │
│ - Validate Range │
│ - Timestamp      │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Store Raw Data   │
│ - InfluxDB       │
│ - Time Series    │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Feature Extract  │
│ - Rolling Stats  │
│ - Aggregations   │
│ - Anomalies      │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Trigger Actions  │
│ - Alerts         │
│ - Irrigation     │
│ - Predictions    │
└──────────────────┘
```

### 3. Training Data Flow

```
Raw Dataset
       │
       ▼
┌──────────────────┐
│ Data Validation  │
│ - Check Quality  │
│ - Split Train/Val│
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Feature Engineer │
│ - Extract Feat.  │
│ - Augmentation   │
│ - Normalize      │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Train Model      │
│ - Forward Pass   │
│ - Backprop       │
│ - Update Weights │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Evaluate Model   │
│ - Test Set       │
│ - Metrics        │
│ - Confusion Mat  │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Register Model   │
│ - MLflow         │
│ - Version        │
│ - Metrics        │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Deploy Model     │
│ - Production     │
│ - A/B Testing    │
│ - Monitor        │
└──────────────────┘
```

## Data Storage

### PostgreSQL Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    location JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crops table
CREATE TABLE crops (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(100),
    species VARCHAR(100),
    planting_date DATE,
    location JSONB,
    status VARCHAR(50)
);

-- Diagnoses table
CREATE TABLE diagnoses (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    crop_id UUID REFERENCES crops(id),
    image_url VARCHAR(500),
    diagnosis VARCHAR(100),
    confidence DECIMAL(5,4),
    severity VARCHAR(50),
    recommendations JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sensor data table
CREATE TABLE sensor_data (
    id UUID PRIMARY KEY,
    crop_id UUID REFERENCES crops(id),
    sensor_type VARCHAR(50),
    value DECIMAL(10,2),
    unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Neo4j Schema

```cypher
// Disease nodes
CREATE (d:Disease {
    name: "Blast",
    scientific_name: "Magnaporthe oryzae",
    symptoms: ["Bercak oval", "Warna abu-abu"],
    severity_levels: ["low", "medium", "high"]
})

// Crop nodes
CREATE (c:Crop {
    name: "Padi",
    species: "Oryza sativa",
    growing_conditions: "Tropis"
})

// Relationships
CREATE (c)-[:SUSCEPTIBLE_TO]->(d)
CREATE (d)-[:TREATED_BY]->(t:Treatment {name: "Fungisida"})
CREATE (d)-[:FAVORED_BY]->(k:Kondisi {humidity: 85, temperature: 28})
```

### Redis Cache Strategy

```
Key Pattern: diagnosis:{user_id}:{timestamp}
Value: JSON response
TTL: 1 hour

Key Pattern: user:{user_id}:history
Value: List of recent diagnoses
TTL: 24 hours

Key Pattern: model:{model_name}:cache
Value: Model predictions cache
TTL: 5 minutes
```

## Data Retention

| Data Type | Retention Period | Storage |
|-----------|------------------|---------|
| User data | Indefinite | PostgreSQL |
| Images | 1 year | MinIO/S3 |
| Predictions | 6 months | PostgreSQL |
| Sensor data | 3 months | InfluxDB |
| Logs | 1 month | Elasticsearch |
| MLflow runs | Indefinite | MLflow |

## Data Security

- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access Control**: RBAC with JWT tokens
- **Audit Logging**: All data access logged
- **Backup**: Daily automated backups
- **GDPR Compliance**: Data deletion on request
