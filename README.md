# MiniGPT

### Experimental AI Chat Application

MiniGPT is an experimental AI project focused on exploring and building custom artificial intelligence components and bringing them together into an interactive chat application.

The project combines a custom Python/NumPy-based AI system, natural language processing, evidence-based retrieval, and an experimental neural language-model component within a modular architecture.

Rather than depending entirely on external AI services, MiniGPT focuses on understanding, developing, and integrating the underlying components of an AI application.

## Overview

MiniGPT brings together several independently developed components:

- **Custom AI Engine** — A custom-built engine for processing questions, retrieving relevant knowledge, and generating responses.
- **Custom NLP** — Text processing, normalization, intent detection, keyword extraction, and contextual understanding.
- **Reasoning Engine** — Evidence-based retrieval and ranking using a custom BM25-based approach.
- **Neural Model** — An experimental decoder-based language model implemented using Python and NumPy.
- **Frontend** — A dedicated chat interface connected to the Python backend.
- **AI Experiments** — A collection of independent modules supporting the application's capabilities.

## Features

### AI Engine

The core system combines custom language processing, knowledge retrieval, and response composition.

It includes:
- Local knowledge retrieval.
- Context-aware question processing.
- Evidence ranking and matching.
- Question decomposition and focus extraction.
- Custom reasoning and response composition.

### Optional Tools

MiniGPT includes a collection of tools that can extend the application's capabilities.

| Tool | Purpose |
|---|---|
| Calculator | Mathematical calculations |
| Unit Converter | Convert between measurement units |
| Date & Time | Date and time information |
| Date Calculator | Date calculations |
| Scientific Math | Scientific and mathematical operations |
| Text Analyzer | Text analysis |
| JSON Tools | JSON processing utilities |
| Web Search | Retrieve information from the web |
| Document Reader | Search and retrieve information from uploaded documents |
| Memory Context | Maintain relevant conversation and profile context |

**All tools are disabled by default.** They can be enabled individually when needed.

### Web Search

Optional web search functionality powered by a DDGS-based search provider, including dedicated news search.

Search is only triggered when the corresponding tool is enabled.

### Document Understanding

Upload documents and retrieve relevant information from their contents.

Supported formats include:
- PDF
- DOCX
- TXT
- Markdown
- CSV
- HTML

The document engine uses extractive retrieval to find relevant information within uploaded files.

### Memory Context

A custom memory system designed to maintain useful information across conversations.

Includes:
- Persistent user context.
- Per-conversation history.
- Explicit memory commands.
- Conversation recaps.
- Contextual follow-up handling.
- Memory deletion.

Memory is enabled for local use and disabled by default in the included hosted deployment configuration.

## Technology

| Component | Technology |
|---|---|
| Main Language | Python |
| Numerical Computing | NumPy |
| Neural Model | Custom Python/NumPy implementation |
| NLP | Custom NLP components |
| Reasoning | Custom BM25-based retrieval |
| Backend | Python HTTP server |
| Frontend | HTML, CSS, JavaScript |
| Web Search | DDGS |
| Document Processing | Custom document retrieval with PDF support |

## Getting Started

### Requirements

- Python installed.
- Internet connection for installing dependencies and using web search.
- A compatible Windows environment for the included launcher.

### Run on Windows

1. Download or clone this repository.
2. Open the project folder.
3. Run `Run Chat.bat`.
4. The launcher installs the required Python packages and starts the application.
5. Your browser opens at:

   `http://127.0.0.1:8000`

### Manual Setup

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

The local chat interface will be available at `http://127.0.0.1:8000`.

## Project Structure

```text
MiniGPT/
│
├── AI Experiments/
│   ├── Tiny AI Engine/
│   ├── Custom NLP/
│   ├── Custom Reasoning Engine/
│   ├── Web Search/
│   ├── Document Engine/
│   ├── Memory Context Engine/
│   ├── Scientific Math/
│   └── ...
│
├── Frontend/
│   └── Chat Interface
│
├── Run Chat.bat
├── app.py
├── requirements.txt
└── render.yaml
```

## Data & Privacy

MiniGPT supports local document storage and persistent memory.

By default:
- Local memory is stored on the local system.
- Uploaded documents are stored in the application's data directory.
- Optional tools remain disabled until explicitly enabled.
- Web searches use an external search provider when requested.
- The included Render configuration disables document uploads and persistent memory for hosted deployments.

Review deployment settings before enabling features that store user information on a hosted server.

## Project Status

**Experimental Development**

MiniGPT is an evolving project focused on exploring custom AI architectures, improving language processing, and developing a modular AI application.

Its components and capabilities may change as development progresses.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

*Exploring artificial intelligence by building its components from the ground up.*
