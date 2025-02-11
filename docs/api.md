# API Reference

This document provides detailed information about the Code Quality Analyzer's FastAPI endpoints, request/response models, and usage examples.

## Base URL

When running locally, the API is available at:
```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for local usage.

## Endpoints

### Analysis Endpoint

```
POST /analyze
```

Initiates code analysis for a GitHub repository.

#### Request Body

```json
{
  "repoUrl": "string",
  "options": {
    "verbose": false,
    "generatePdf": true,
    "exportJson": true,
    "minLines": 6,
    "comparison": {
      "enabled": false,
      "type": "string",
      "options": {},
      "metrics": []
    },
    "includeTestAnalysis": true,
    "includeTrends": true,
    "mlAnalysis": {
      "enabled": true,
      "teamAvailability": {
        "additionalProp1": [
          ["string", "string"]
        ]
      },
      "analyzeDeploymentWindows": true,
      "analyzeResourceRequirements": true,
      "analyzeRollbackRisks": true,
      "analyzeIncidentPrediction": true,
      "generateHistoricalTrends": true
    }
  }
}
```

#### Options Description

- `verbose` (boolean): Enable detailed logging
- `generatePdf` (boolean): Generate PDF report
- `exportJson` (boolean): Export metrics as JSON
- `minLines` (integer): Minimum lines for duplicate code detection
- `includeTestAnalysis` (boolean): Include test coverage analysis
- `includeTrends` (boolean): Include historical trend analysis

#### Comparison Options

- `enabled` (boolean): Enable comparative analysis
- `type` (string): Type of comparison ("directory", "snapshot", "historical")
- `options` (object): Comparison-specific options
- `metrics` (array): Metrics to compare

##### Directory Comparison Options
```json
{
  "dir1": "string",
  "dir2": "string"
}
```

##### Snapshot Comparison Options
```json
{
  "baseline": "string",
  "target": "string",
  "component": "string"
}
```

##### Historical Comparison Options
```json
{
  "startDate": "string",
  "endDate": "string",
  "component": "string",
  "interval": "string"
}
```

#### ML Analysis Options

- `enabled` (boolean): Enable ML-based analysis
- `teamAvailability` (object): Team working hours
- `analyzeDeploymentWindows` (boolean): Analyze optimal deployment times
- `analyzeResourceRequirements` (boolean): Analyze resource needs
- `analyzeRollbackRisks` (boolean): Analyze rollback risks
- `analyzeIncidentPrediction` (boolean): Predict potential incidents
- `generateHistoricalTrends` (boolean): Generate historical trends

#### Response

```json
{
  "success": true,
  "outputDir": "string",
  "files": {},
  "comparison": {
    "type": "string",
    "differences": {},
    "timeline": [],
    "metrics": {}
  },
  "mlAnalysis": {
    "optimalWindows": [
      {
        "startTime": "string",
        "endTime": "string",
        "riskScore": 0,
        "teamAvailability": 0
      }
    ],
    "resourcePrediction": {
      "recommendedTeamSize": 0,
      "estimatedSupportDuration": 0,
      "requiredSkills": []
    },
    "rollbackPrediction": {
      "probability": 0,
      "severityLevel": "string",
      "riskFactors": {}
    },
    "confidenceScores": {}
  },
  "pdfPath": "string",
  "jsonPath": "string",
  "logPath": "string"
}
```

### Download Endpoint

```
GET /download/{file_path}
```

Downloads generated files (PDF reports, JSON exports, etc.).

#### Parameters

- `file_path` (string, path): Path to the file to download

#### Response

File download response with appropriate content type.

### Root Endpoint

```
GET /
```

Serves the web interface.

#### Response

HTML content of the web interface.

## Examples

### Basic Analysis Request

```python
import aiohttp
import asyncio

async def analyze_repo():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8000/analyze',
            json={
                'repoUrl': 'https://github.com/username/repo',
                'options': {
                    'generatePdf': True,
                    'exportJson': True
                }
            }
        ) as response:
            result = await response.json()
            print(result)

asyncio.run(analyze_repo())
```

### Comparative Analysis Request

```python
comparison_request = {
    'repoUrl': 'https://github.com/username/repo',
    'options': {
        'generatePdf': True,
        'comparison': {
            'enabled': True,
            'type': 'snapshot',
            'options': {
                'baseline': 'main~1',
                'target': 'main',
                'component': 'src/core'
            },
            'metrics': ['complexity', 'security']
        }
    }
}
```

### ML Analysis Request

```python
ml_request = {
    'repoUrl': 'https://github.com/username/repo',
    'options': {
        'generatePdf': True,
        'mlAnalysis': {
            'enabled': True,
            'teamAvailability': {
                'developer1': [['09:00', '17:00']],
                'developer2': [['10:00', '18:00']]
            },
            'analyzeDeploymentWindows': True,
            'analyzeResourceRequirements': True
        }
    }
}
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages:

- 200: Successful operation
- 400: Bad request (invalid parameters)
- 404: Resource not found
- 500: Server error

Error Response Format:
```json
{
  "detail": "Error message describing the issue"
}
```

## Rate Limiting

Currently, there are no rate limits implemented for local usage.

## WebSocket Support

The API includes WebSocket support for real-time progress updates during analysis:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis');
ws.onmessage = (event) => {
    const progress = JSON.parse(event.data);
    console.log('Analysis progress:', progress);
};
```

## Using with CORS

The API supports CORS for all origins by default. Custom CORS configuration can be implemented if needed.