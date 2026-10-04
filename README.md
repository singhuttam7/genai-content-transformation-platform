# Context2Artifact --- Gen AI Platform for Automated Content Transformation

> **One Source. Understood in Context. Transformed into Purpose-Specific
> Artifacts.**

Context2Artifact is a production-oriented GenAI platform developed for
**Smart India Hackathon 2026 --- SIH26154: Gen AI Platform for Automated
Content Transformation**.

It transforms source information such as text, prompts, documents and
URLs into purpose-specific communication artifacts through a
configurable pipeline combining ingestion, extraction, normalization,
knowledge retrieval, agentic orchestration, LLM generation, validation,
provenance and artifact persistence.

------------------------------------------------------------------------

## 1. Project Information

  Field                   Value
  ----------------------- ------------------------------------------------------
  Product                 Context2Artifact
  SIH Problem Statement   SIH26154
  Problem                 Gen AI Platform for Automated Content Transformation
  Theme                   Blockchain & Cybersecurity
  Category                Software
  Team ID                 121654
  Team                    Gamma Coders

------------------------------------------------------------------------

## 2. Problem

Organizations frequently need to transform the same information into
different communication formats. A report, article, advisory or research
document may need to become an executive summary, advisory, social-media
communication, presentation, infographic or other audience-specific
artifact.

Manual transformation creates:

-   repetitive rewriting,
-   context loss,
-   inconsistent messaging,
-   repeated formatting work,
-   difficulty adapting information for different audiences,
-   fragmented transformation workflows.

Context2Artifact addresses this by converting the
source-to-communication process into a reusable AI workflow.

------------------------------------------------------------------------

## 3. Solution

Context2Artifact provides a unified transformation platform where an
operator can:

1.  provide source information,
2.  extract and normalize its content,
3.  retrieve relevant context,
4.  configure audience, tone, language, detail and communication
    objectives where supported,
5.  execute a reusable transformation workflow,
6.  generate a purpose-specific artifact,
7.  validate the result,
8.  persist execution and artifact information.

The platform focuses on **controlled transformation rather than
unrestricted free-form generation**.

### Core differentiators

-   Transformation, not simple generation
-   One source → multiple communication purposes
-   Context + configuration
-   Reusable workflows
-   Agentic orchestration
-   Retrieval-Augmented Generation
-   Validation and provenance-oriented processing
-   Provider-independent LLM architecture
-   Modular ingestion and artifact architecture

------------------------------------------------------------------------

## 4. Core Workflow

``` text
SOURCE
  ↓
INGESTION
  ↓
EXTRACTION + NORMALIZATION
  ↓
KNOWLEDGE / CONTEXT
  ↓
WORKFLOW ORCHESTRATION
  ↓
GENAI TRANSFORMATION
  ↓
ARTIFACT GENERATION
  ↓
VALIDATION + PROVENANCE
  ↓
FINAL DELIVERY
```

### Source ingestion

The platform accepts text, prompts, documents and URLs, with an
extensible multimodal processing foundation.

### Extraction and normalization

Source-specific processors convert raw inputs into a consistent internal
representation.

### Knowledge and context

The RAG subsystem can chunk content, generate embeddings, store vectors
and retrieve relevant context.

### Orchestration

A workflow determines transformation requirements, configuration and
agent execution.

### Transformation

The selected transformation agent uses contextual information and the
LLM gateway to generate the requested artifact.

### Validation and delivery

Generated outputs are validated and persisted for review and downstream
use.

------------------------------------------------------------------------

## 5. Main Features

### Source processing

-   Text ingestion
-   Prompt ingestion
-   PDF processing
-   DOCX processing
-   HTML processing
-   URL ingestion
-   OCR-enabled processing
-   Audio/video processing foundation
-   Vision processing foundation

### AI and knowledge

-   RAG
-   Sentence Transformer embeddings
-   PostgreSQL + pgvector
-   Context retrieval
-   LangChain
-   LangGraph
-   Provider-independent LLM gateway
-   Groq/Gemini provider adapters
-   Optional Tavily external knowledge retrieval

### Transformation

The transformation architecture supports dedicated agents for
communication purposes such as:

-   Executive Summary
-   Advisory
-   Social Media
-   Infographic
-   Presentation
-   Video-oriented transformation

The architecture is extensible for additional artifact generators.

### Platform

-   Workflow creation
-   Execution tracking
-   Agent sequencing
-   Artifact persistence
-   Source persistence
-   Validation architecture
-   Provenance architecture
-   Configurable integrations

------------------------------------------------------------------------

## 6. Supported Inputs and Outputs

### Demonstrated / active local test path

``` text
TEXT
PROMPT
URL
PDF
```

### Processing foundation

-   DOCX
-   HTML
-   Images
-   Audio/video
-   OCR
-   ASR
-   Vision

The exact modality available depends on deployment configuration and
enabled services.

### Output examples

-   Executive summaries
-   Advisories
-   Social-media content
-   Infographics
-   Presentations
-   Video-oriented artifacts
-   Additional extensible communication artifacts

------------------------------------------------------------------------

## 7. System Architecture

``` text
                    ┌──────────────────────┐
                    │    React + Vite      │
                    │      Frontend        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI Backend   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
       │  Ingestion  │ │  Workflow /  │ │   Artifact   │
       │  & Parsing  │ │  Execution   │ │  Management  │
       └──────┬──────┘ └──────┬───────┘ └──────────────┘
              │               │
              ▼               ▼
       ┌─────────────┐ ┌──────────────┐
       │ Normalize   │ │ Transformation│
       │ + Extract   │ │    Agents     │
       └──────┬──────┘ └──────┬───────┘
              │               │
              └───────┬───────┘
                      ▼
             ┌──────────────────┐
             │   RAG / Context  │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Sentence         │
             │ Transformers     │
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ PostgreSQL +     │
             │ pgvector         │
             └──────────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ LLM Gateway      │
             │ Provider Adapters│
             └────────┬─────────┘
                      ▼
             ┌──────────────────┐
             │ Generated        │
             │ Artifact         │
             └──────────────────┘
```

------------------------------------------------------------------------

## 8. Technology Stack

  Layer                 Technology
  --------------------- ------------------------------
  Frontend              React 19
  Tooling               Vite 7
  Backend               Python + FastAPI
  Server                Uvicorn
  ORM                   SQLAlchemy 2.x
  Migrations            Alembic
  Database              PostgreSQL
  Vector search         pgvector
  Embeddings            Sentence Transformers
  Embedding model       all-MiniLM-L6-v2
  AI orchestration      LangChain
  Workflow/agents       LangGraph
  LLM layer             Provider-independent gateway
  Providers             Groq / Gemini adapters
  External search       Tavily
  PDF                   pypdf
  DOCX                  python-docx
  HTML                  BeautifulSoup
  OCR                   Tesseract
  ASR                   Faster-Whisper
  Media                 FFmpeg / PyAV
  Local AI runtime      Ollama
  Production frontend   Nginx
  Containerization      Docker / Docker Compose
  Cloud prototype       Render
  Cloud database        Neon PostgreSQL

------------------------------------------------------------------------

## 9. Repository Structure

``` text
genai-content-transformation-platform/
├── .github/
│   └── workflows/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   ├── artifacts/
│   │   ├── ingestion/
│   │   ├── integrations/
│   │   ├── intelligence/
│   │   ├── memory/
│   │   ├── models/
│   │   ├── models_ai/
│   │   ├── observability/
│   │   ├── orchestration/
│   │   ├── provenance/
│   │   ├── rag/
│   │   ├── security/
│   │   ├── services/
│   │   ├── storage/
│   │   ├── tools/
│   │   ├── validation/
│   │   └── workers/
│   ├── alembic/
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── alembic.ini
├── frontend/
│   ├── src/
│   │   └── features/
│   │       ├── agents/
│   │       ├── artifacts/
│   │       ├── dashboard/
│   │       ├── executions/
│   │       ├── integrations/
│   │       ├── knowledge/
│   │       ├── settings/
│   │       ├── source-ingestion/
│   │       ├── transformation/
│   │       └── workflow-builder/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── data/
├── infrastructure/
├── scripts/
├── docker-compose.yml
├── render.yaml
├── .env.example
├── .dockerignore
└── README.md
```

------------------------------------------------------------------------

## 10. Local Development

### Prerequisites

-   Python 3.13+
-   Node.js 22+
-   npm
-   PostgreSQL 18 + pgvector, or Docker
-   Git
-   Docker Desktop for containerized development
-   Tesseract for OCR when required
-   FFmpeg for media processing when required

### Backend setup

``` powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure environment variables, then run:

``` powershell
alembic upgrade head
uvicorn app.main:app --reload
```

Backend:

``` text
http://127.0.0.1:8000
```

Swagger:

``` text
http://127.0.0.1:8000/docs
```

Health:

``` text
http://127.0.0.1:8000/api/v1/health
```

> **Important:** From the `backend` directory the correct Uvicorn target
> is `app.main:app`, not `main:app`.

### Frontend setup

``` powershell
cd frontend
npm ci
npm run dev
```

Frontend:

``` text
http://localhost:5173
```

Production build:

``` powershell
npm run build
```

------------------------------------------------------------------------

## 11. Environment Configuration

Use `.env.example` as the safe template.

Typical configuration:

``` env
ENVIRONMENT=development
DEBUG=true

LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=

DATABASE_URL=

FRONTEND_URL=http://localhost:5173

STORAGE_BACKEND=local
STORAGE_ROOT=./data/storage

SECRET_KEY=

OCR_ENABLED=true
OCR_PROVIDER=tesseract
OCR_DEFAULT_LANGUAGE=eng

ASR_ENABLED=true
ASR_PROVIDER=faster_whisper
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

VISION_ENABLED=true
VISION_PROVIDER=local
VISION_MODEL=gemma3:4b
VISION_BASE_URL=http://localhost:11434
```

Do not commit real credentials.

------------------------------------------------------------------------

## 12. Docker

The Docker Compose stack contains:

-   PostgreSQL + pgvector
-   Ollama
-   FastAPI backend
-   React/Vite frontend served through Nginx

Start infrastructure:

``` powershell
docker compose up -d postgres ollama
```

Start the complete application:

``` powershell
docker compose up -d
```

Check services:

``` powershell
docker compose ps
```

Backend logs:

``` powershell
docker compose logs -f backend
```

Frontend logs:

``` powershell
docker compose logs -f frontend
```

Stop:

``` powershell
docker compose down
```

The production frontend Nginx server proxies `/api/` requests to the
backend service.

------------------------------------------------------------------------

## 13. Database and Migrations

The backend uses asynchronous SQLAlchemy with PostgreSQL.

Run migrations:

``` powershell
cd backend
alembic upgrade head
```

View migration history:

``` powershell
alembic history
```

Create a migration:

``` powershell
alembic revision --autogenerate -m "describe change"
```

Always review generated migrations before applying them.

The database supports source, knowledge, transformation, execution and
artifact-related persistence.

------------------------------------------------------------------------

## 14. RAG and Embeddings

Current embedding model:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

``` text
384
```

RAG flow:

``` text
Source
  ↓
Chunking
  ↓
Sentence Transformer
  ↓
Embedding
  ↓
pgvector
  ↓
Similarity Retrieval
  ↓
Relevant Context
  ↓
Transformation Agent
  ↓
LLM
```

RAG is designed to reduce context loss by retrieving relevant source
information before transformation.

### Resource consideration

Sentence Transformers/PyTorch can require significant memory. The free
Render backend has a 512 MB memory constraint and is therefore not
currently considered reliable for the complete local RAG execution path.
The local/Docker environment is the primary validated environment for
full RAG execution.

------------------------------------------------------------------------

## 15. OCR, ASR and Vision

### OCR

Tesseract is used for OCR-enabled processing such as scanned documents
and images.

### ASR

Faster-Whisper provides speech-to-text processing for supported
audio/video workflows.

Example configuration:

``` env
ASR_ENABLED=true
ASR_PROVIDER=faster_whisper
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

### Vision

Local vision processing can use Ollama.

Example:

``` env
VISION_ENABLED=true
VISION_PROVIDER=local
VISION_MODEL=gemma3:4b
VISION_BASE_URL=http://localhost:11434
```

Vision can be disabled for resource-constrained deployments.

------------------------------------------------------------------------

## 16. Agent and Workflow Architecture

The execution architecture follows:

``` text
Workflow
   ↓
Execution Service
   ↓
Workflow Executor
   ↓
Agent Registry
   ↓
Transformation Agent
   ↓
RAG Context
   ↓
LLM Gateway
   ↓
Agent Result
   ↓
Artifact
```

Transformation agents are separated by communication purpose.

This provides:

-   separation of concerns,
-   reusable execution infrastructure,
-   easier testing,
-   extensibility,
-   workflow-level control.

The agent integration layer also supports explicit RAG enable/disable
behavior.

------------------------------------------------------------------------

## 17. Testing

The project uses an incremental testing strategy:

``` text
Focused Tests
     ↓
Subsystem Tests
     ↓
Full Regression
     ↓
Manual / Deployment Verification
```

Latest verified local regression baseline:

``` text
2992 passed
1 warning
```

Run all backend tests:

``` powershell
cd backend
pytest -q
```

Run a test file:

``` powershell
pytest tests/path/to/test_file.py -q
```

Run one test:

``` powershell
pytest tests/path/to/test_file.py::test_name -q
```

### Development rule

After every meaningful code change:

1.  Run focused tests.
2.  Run affected subsystem tests.
3.  Run the full regression suite.
4.  Verify the application if the change affects runtime behavior.
5.  Commit only after validation.

------------------------------------------------------------------------

## 18. Deployment

### Local/Docker

Docker is the primary reproducible development/runtime option.

### Render + Neon

The project has a cloud prototype consisting of:

``` text
React/Vite frontend
        ↓
Render Static Site

FastAPI backend
        ↓
Render Web Service

PostgreSQL
        ↓
Neon
```

The backend uses SSL-enabled async PostgreSQL connectivity.

### Cloud resource limitation

The current free Render backend has 512 MB RAM. Full RAG execution using
the local Sentence Transformer/PyTorch stack is not considered reliable
within that constraint.

A production deployment should use adequate compute or a
resource-optimized RAG architecture.

------------------------------------------------------------------------

## 19. Current Prototype Status

### Verified locally

-   React/Vite frontend
-   FastAPI backend
-   PostgreSQL
-   pgvector
-   Alembic migrations
-   source ingestion
-   text processing
-   PDF processing
-   URL ingestion
-   transformation creation
-   workflow creation
-   execution management
-   agent orchestration
-   artifact generation and persistence
-   RAG architecture
-   Sentence Transformer adapter
-   OCR infrastructure
-   ASR infrastructure
-   Vision integration
-   Docker runtime
-   frontend/backend integration
-   automated regression suite

### Recommended local demonstration

``` text
Text → Prompt → URL → PDF
```

A small PDF can be used to validate the PDF ingestion and transformation
path.

### Current deployment status

The cloud prototype is deployed, but the free-tier memory limitation
means the complete local RAG execution path should be demonstrated
locally/Docker rather than represented as fully validated production RAG
execution.

------------------------------------------------------------------------

## 20. Known Limitations

1.  Embedding and AI models can require substantial memory.
2.  Free-tier cloud resources constrain RAG execution.
3.  Full distributed/background processing is future scale-up work.
4.  Multimodal processing infrastructure exists, but not every modality
    has equal end-to-end demonstration maturity.
5.  Performance percentages should only be published after controlled
    benchmarking.
6.  Production authentication and enterprise security hardening require
    additional work.
7.  Advanced observability and distributed execution remain future
    scale-up areas.

------------------------------------------------------------------------

## 21. Production Security

Never commit:

``` text
.env
API keys
database passwords
production SECRET_KEY
private certificates
provider credentials
```

Security hardening should include:

-   authentication and authorization,
-   strict CORS,
-   upload validation,
-   file-size limits,
-   URL/SSRF protection,
-   safe redirects,
-   content-type validation,
-   rate limiting,
-   secure secret management,
-   structured audit logging.

External URL ingestion should be protected against access to
private/internal networks and malicious or oversized responses.

------------------------------------------------------------------------

## 22. Future Roadmap

### Production execution

-   Background workers
-   Queue-based processing
-   Resource-aware model loading
-   Better observability
-   Metrics
-   Distributed tracing
-   Improved recovery

### AI expansion

-   Additional LLM providers
-   Additional transformation agents
-   Improved multimodal understanding
-   Advanced retrieval
-   Stronger validation

### Enterprise

-   Enterprise document repositories
-   Organizational knowledge bases
-   Communication-platform integrations
-   Content management systems
-   Secure internal data sources
-   Enterprise identity/access control

### Large-scale deployment

``` text
Load Balancer
     ↓
API Services
     ↓
Job Queue
     ├── Ingestion Workers
     ├── OCR Workers
     ├── ASR Workers
     ├── Embedding Workers
     └── Transformation Workers
     ↓
PostgreSQL + pgvector
     +
Object Storage
```

Redis/Celery are reserved for future asynchronous scaling and are not
required for the current runtime.

------------------------------------------------------------------------

## 23. SIH Information

**Smart India Hackathon 2026**

-   Problem Statement: **SIH26154**
-   Problem: **Gen AI Platform for Automated Content Transformation**
-   Theme: **Blockchain & Cybersecurity**
-   Category: **Software**
-   Team ID: **121654**
-   Team: **Gamma Coders**
-   Product: **Context2Artifact**

### Product statement

> **One Source. Understood in Context. Transformed into Purpose-Specific
> Artifacts.**

------------------------------------------------------------------------

## 24. Research Foundation

### Retrieval-Augmented Generation

Lewis et al., 2020.

Provides the research foundation for grounding generation with retrieved
knowledge.

### ReAct

Yao et al., 2022.

Relevant to reasoning and action-oriented agentic workflows.

### Lost in the Middle

Liu et al., 2023.

Relevant to context selection and retrieval for long source material.

### Sentence Transformers

Used for semantic embeddings and retrieval.

### pgvector

Used for vector similarity search integrated with PostgreSQL.

------------------------------------------------------------------------

## 25. Project Links

### GitHub

https://github.com/singhuttam7/genai-content-transformation-platform

### Prototype

https://genai-content-transformation-platform-1.onrender.com

------------------------------------------------------------------------

## 26. Development Guidelines

The project follows an incremental production-grade workflow:

``` text
Plan subsystem
     ↓
Implement smallest safe change
     ↓
Focused tests
     ↓
Subsystem tests
     ↓
Full regression
     ↓
Manual/runtime verification
     ↓
Document
     ↓
Commit
     ↓
Move to next subsystem
```

Avoid unnecessary architectural changes to already-validated components.

------------------------------------------------------------------------

## 27. Final Vision

Context2Artifact is designed to move Generative AI beyond simple prompt-to-text
generation.

The platform focuses on a complete transformation lifecycle:

``` text
INFORMATION
     ↓
UNDERSTANDING
     ↓
CONTEXT
     ↓
CONFIGURATION
     ↓
AGENTIC TRANSFORMATION
     ↓
VALIDATION
     ↓
TRACEABLE ARTIFACT
```

The goal is to make communication transformation:

-   reusable,
-   context-aware,
-   configurable,
-   extensible,
-   traceable,
-   and suitable for future organizational scale.

> **Context2Artifact --- One Source. Understood in Context. Transformed
> into Purpose-Specific Artifacts.**
