# Org SLM Dashboard API

Welcome to the **Local AI Coding Assistant Platform**. This project is a complete, self-hosted, multi-agent AI orchestration system built with Python, FastAPI, and PyTorch. It utilizes **Small Language Models (SLMs)** and **Low-Rank Adaptation (LoRA)** to dynamically generate, plan, and execute code across a massive variety of frontend and backend frameworks.

## 🏗 System Architecture

The core philosophy of this project is high-performance, low-resource autonomous coding. Instead of loading giant monolithic LLMs, the system loads a **single lightweight base model** into memory and dynamically injects tiny domain-specific LoRA adapters (e.g., React, Flutter, Django) on the fly based on the user's task.

### 1. The Orchestrator & Server (`server.py`)
A FastAPI backend that serves as the command center for the desktop app. It exposes several endpoints:
- **`POST /generate`**: Receives a task and preferred stacks. Uses the Orchestrator to classify the request, injects the proper LoRA adapters, generates the code, and extracts file paths.
- **`POST /plan`**: Acts as a software architect. Analyzes the task and outputs a detailed, Markdown-formatted implementation plan using standard GitHub formatting.
- **`POST /execute`**: Takes a generated code plan and physically writes the files into the workspace, or executes setup terminal commands (like `npm install`).
- **`POST /verify`**: Generates a QA walkthrough describing what was built and how the user can test it locally.
- **`POST /train/{domain}`**: Triggers asynchronous background training to fine-tune a specific domain's adapter.

### 2. The AI Pipeline (`pipeline.py`)
The `OrgSLMPipeline` handles the heavy lifting of the AI generation:
- Loads `checkpoints/base_pretrained.pt`.
- Manages an `AdapterManager` to hot-swap `.pt` LoRA adapters into the network.
- Uses an `Orchestrator` to decompose complex full-stack requests (e.g., "build a React native login with a Django backend") into sub-tasks.
- Uses a `MergerAgent` to combine multiple domain outputs into cohesive solutions.
- Integrates a **Local RAG Engine** (`rag_engine.py`) to index the user's workspace and provide context to the SLM before generation.

### 3. The Desktop Client (`desktop-app/`)
The primary interface for the user. Built using web technologies (HTML/JS/CSS) packaged as a desktop application (e.g., via Electron). It communicates directly with the FastAPI endpoints to request code generation, planning, and execution without relying on a web browser.

### 4. Evaluation Harness (`harness/`)
A dedicated benchmarking suite designed to ensure the local SLMs are performing accurately:
- **`evaluate.py`**: Runs the pipeline against a JSON list of benchmark tasks. It measures latency, orchestrator routing accuracy, and whether the generated code is syntactically valid (AST checks).
- **`report_generator.py`**: Parses the evaluation outputs and produces a detailed, Markdown-formatted QA report (`evaluation_report.md`).

## 🚀 Getting Started

### 1. Bootstrapping the Base Model
Before starting the server or running evaluations, you must initialize the base model weights:
```bash
python bootstrap_base.py
```
This will create the `checkpoints/base_pretrained.pt` file.

### 2. Running the Server
Start the central FastAPI orchestrator:
```bash
python server.py
```
The server will run on `http://0.0.0.0:8080`.

### 3. Running the Desktop App
Navigate to the `desktop-app` directory, install dependencies, and launch:
```bash
cd desktop-app
npm install
npm start
```

### 4. Running the Evaluation Suite
To benchmark your local models against the test dataset:
```bash
python harness/evaluate.py
python harness/report_generator.py
```

## 🧠 Supported Domains (LoRA Adapters)
The orchestrator currently supports hot-swapping adapters for the following stacks:
- **Frontend**: React, React Native, Flutter, Angular, Next.js
- **Backend**: Python Backend (Django), Python FastAPI, Java SpringBoot, JS Express, .NET Web API
- **Specialty**: AI/ML Pipelines, Merger Agent (for combining code)

## 🛠 Model Training
If you want to improve the system's performance on a specific stack, you can trigger asynchronous training via the API:
```bash
curl -X POST http://localhost:8080/train/react
```
This triggers `train_adapters.py` in the background, fine-tuning the `react_lora.pt` adapter without taking the server offline. When finished, the pipeline automatically reloads the new weights.
