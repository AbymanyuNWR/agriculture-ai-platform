# API Specification

## Overview

REST API Specification untuk Agriculture AI Platform.

## Base URL

```
Production: https://api.agriculture-ai.com
Staging: https://staging-api.agriculture-ai.com
Development: http://localhost:8000
```

## Authentication

Semua endpoint memerlukan JWT token kecuali `/auth/login` dan `/auth/register`.

```
Authorization: Bearer <token>
```

## Endpoints

### Authentication

#### POST /api/v1/auth/register

Register user baru.

**Request:**
```json
{
  "name": "Budi Santoso",
  "email": "budi@example.com",
  "password": "password123",
  "phone": "+6281234567890",
  "location": {
    "lat": -6.2088,
    "lng": 106.8456,
    "address": "Jakarta, Indonesia"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "name": "Budi Santoso",
    "email": "budi@example.com",
    "token": "jwt_token"
  }
}
```

#### POST /api/v1/auth/login

Login user.

**Request:**
```json
{
  "email": "budi@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "name": "Budi Santoso",
    "email": "budi@example.com",
    "token": "jwt_token"
  }
}
```

### Diagnosis

#### POST /api/v1/diagnosis/predict

Upload gambar dan dapatkan diagnosis penyakit.

**Request:**
```
Content-Type: multipart/form-data

image: <file>
crop_type: padi (optional)
location: {"lat": -6.2088, "lng": 106.8456} (optional)
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "crop_type": "padi",
    "diagnosis": "blast",
    "confidence": 0.92,
    "severity": "high",
    "symptoms": [
      "Bercak oval pada daun",
      "Warna abu-abu kecoklatan",
      "Pinggiran bercak tidak beraturan"
    ],
    "causes": [
      "Jamur Magnaporthe oryzae",
      "Kelembapan tinggi (>85%)"
    ],
    "recommendations": [
      "Segera semprot dengan fungisida Tricyclazole",
      "Dosis: 2ml per liter air",
      "Ulangi semprot setelah 7 hari"
    ],
    "treatment_options": [
      {
        "name": "Tricyclazole",
        "dosage": "2ml/liter",
        "application": "Semprot",
        "frequency": "2 kali seminggu"
      }
    ],
    "prevention_tips": [
      "Gunakan varietas tahan",
      "Atur jarak tanam",
      "Jaga kebersihan lahan"
    ],
    "environmental_factors": {
      "humidity": 85,
      "temperature": 28,
      "rainfall": 40
    },
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

#### GET /api/v1/diagnosis/history

Dapatkan riwayat diagnosis user.

**Query Parameters:**
- `page` (integer, default: 1)
- `limit` (integer, default: 10)

**Response:**
```json
{
  "status": "success",
  "data": {
    "diagnoses": [
      {
        "id": "uuid",
        "crop_type": "padi",
        "diagnosis": "blast",
        "confidence": 0.92,
        "created_at": "2024-01-15T10:30:00Z"
      }
    ],
    "total": 25,
    "page": 1,
    "limit": 10
  }
}
```

#### GET /api/v1/diagnosis/{id}

Dapatkan detail diagnosis.

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "crop_type": "padi",
    "diagnosis": "blast",
    "confidence": 0.92,
    "severity": "high",
    "symptoms": [...],
    "causes": [...],
    "recommendations": [...],
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Crops

#### GET /api/v1/crops

Dapatkan daftar tanaman user.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "uuid",
      "name": "Sawah Blok A",
      "species": "padi",
      "planting_date": "2024-01-01",
      "location": {
        "lat": -6.2088,
        "lng": 106.8456
      },
      "status": "healthy"
    }
  ]
}
```

#### POST /api/v1/crops

Tambah tanaman baru.

**Request:**
```json
{
  "name": "Sawah Blok A",
  "species": "padi",
  "planting_date": "2024-01-01",
  "location": {
    "lat": -6.2088,
    "lng": 106.8456
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "name": "Sawah Blok A",
    "species": "padi",
    "planting_date": "2024-01-01",
    "status": "healthy"
  }
}
```

### Sensors

#### POST /api/v1/sensors/data

Kirim data sensor.

**Request:**
```json
{
  "crop_id": "uuid",
  "sensor_type": "temperature",
  "value": 28.5,
  "unit": "celsius",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "recorded_at": "2024-01-15T10:30:00Z"
  }
}
```

#### GET /api/v1/sensors/data/{crop_id}

Dapatkan data sensor untuk tanaman.

**Query Parameters:**
- `sensor_type` (string, optional)
- `start_date` (string, optional)
- `end_date` (string, optional)

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "uuid",
      "sensor_type": "temperature",
      "value": 28.5,
      "unit": "celsius",
      "recorded_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Knowledge

#### GET /api/v1/knowledge/diseases

Dapatkan daftar penyakit.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "uuid",
      "name": "Blast",
      "scientific_name": "Magnaporthe oryzae",
      "crops": ["padi"],
      "symptoms": ["Bercak oval", "Warna abu-abu"],
      "severity_levels": ["low", "medium", "high"]
    }
  ]
}
```

#### GET /api/v1/knowledge/diseases/{id}

Dapatkan detail penyakit.

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "uuid",
    "name": "Blast",
    "scientific_name": "Magnaporthe oryzae",
    "description": "Penyakit jamur yang menyerang padi",
    "symptoms": [
      "Bercak oval pada daun",
      "Warna abu-abu kecoklatan",
      "Pinggiran bercak tidak beraturan"
    ],
    "causes": [
      "Jamur Magnaporthe oryzae",
      "Kelembapan tinggi (>85%)",
      "Suhu 25-30°C"
    ],
    "treatments": [
      {
        "name": "Tricyclazole",
        "dosage": "2ml/liter",
        "application": "Semprot"
      }
    ],
    "prevention": [
      "Gunakan varietas tahan",
      "Atur jarak tanam",
      "Jaga kebersihan lahan"
    ],
    "environmental_factors": {
      "humidity": 85,
      "temperature": 28,
      "rainfall": 40
    }
  }
}
```

### Chat (LLM)

#### POST /api/v1/chat

Konsultasi dengan AI.

**Request:**
```json
{
  "message": "Daun padi saya kuning, kenapa?",
  "context": {
    "crop_type": "padi",
    "location": {
      "lat": -6.2088,
      "lng": 106.8456
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "response": "Daun padi yang menguning bisa disebabkan oleh beberapa faktor:\n\n1. **Kekurangan Nitrogen**: Gejala berupa daun kuning merata dari ujung ke bawah.\n   - Solusi: Berikan pupuk urea 50g per tanaman\n\n2. **Serangan Wereng**: Daun menguning dan menggulung.\n   - Solusi: Semprot insektisida imidakloprid\n\n3. **Kelembapan Berlebihan**: Akar busuk karena air menggenang.\n   - Solusi: Perbaiki drainase\n\nBerdasarkan lokasi Anda di Jakarta, kemungkinan besar disebabkan oleh kelembapan tinggi. Periksa akar tanaman untuk memastikan.",
    "confidence": 0.85,
    "sources": [
      "Knowledge Graph: Rice Diseases",
      "Weather Data: High humidity alert"
    ]
  }
}
```

### Weather

#### GET /api/v1/weather/{location}

Dapatkan data cuaca.

**Query Parameters:**
- `lat` (number)
- `lng` (number)

**Response:**
```json
{
  "status": "success",
  "data": {
    "current": {
      "temperature": 28,
      "humidity": 75,
      "rainfall": 10,
      "wind_speed": 5,
      "condition": "berawan"
    },
    "forecast": [
      {
        "date": "2024-01-16",
        "temperature": 29,
        "humidity": 70,
        "rainfall": 0
      }
    ],
    "alerts": [
      {
        "type": "rain",
        "message": "Hujan ringan dipredesikan besok",
        "severity": "low"
      }
    ]
  }
}
```

## Error Responses

### 400 Bad Request

```json
{
  "status": "error",
  "message": "Invalid input",
  "errors": [
    {
      "field": "email",
      "message": "Invalid email format"
    }
  ]
}
```

### 401 Unauthorized

```json
{
  "status": "error",
  "message": "Invalid or expired token"
}
```

### 403 Forbidden

```json
{
  "status": "error",
  "message": "Insufficient permissions"
}
```

### 404 Not Found

```json
{
  "status": "error",
  "message": "Resource not found"
}
```

### 429 Too Many Requests

```json
{
  "status": "error",
  "message": "Rate limit exceeded",
  "retry_after": 60
}
```

### 500 Internal Server Error

```json
{
  "status": "error",
  "message": "Internal server error"
}
```

## Rate Limits

| Endpoint | Limit (per minute) |
|----------|-------------------|
| /auth/* | 10 |
| /diagnosis/* | 30 |
| /crops/* | 60 |
| /sensors/* | 100 |
| /knowledge/* | 60 |
| /chat | 20 |
| /weather/* | 60 |

## Versioning

API versioning via URL path: `/api/v1/`, `/api/v2/`

## Pagination

Semua list endpoints mendukung pagination:

```
GET /api/v1/diagnosis/history?page=1&limit=10
```

Response includes:
- `total`: Total items
- `page`: Current page
- `limit`: Items per page
- `has_next`: Boolean
- `has_prev`: Boolean
