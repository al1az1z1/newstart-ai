# NewStart AI – Document Routing

NewStart AI is an agentic AI platform designed to help newcomers understand official government documents by automatically identifying the document category and routing it to the appropriate AI service agent.

The project compares three document-routing approaches:

* Fine-tuned BERT Base
* LLM Agent (without RAG)
* LLM Agent (with RAG)

The goal is to evaluate which approach provides the best document classification performance for routing government documents such as USCIS, DMV, Social Security Administration (SSA), and IRS forms and notices.

## Data Collection

The dataset is built from publicly available government documents collected from official agency websites. Crawlers are used to automatically download forms, instructions, and other public documents from sources such as:

* U.S. Citizenship and Immigration Services (USCIS)
* California Department of Motor Vehicles (DMV)
* Social Security Administration (SSA)
* Internal Revenue Service (IRS)

**Note:** Using crawlers ensures that the dataset can be collected efficiently, reproducibly, and updated automatically as new public documents become available. Only publicly accessible documents intended for public distribution are collected.

The downloaded documents are organized, labeled, and prepared for training and evaluation.

## Project Structure

* **Data Collection** – Crawlers and dataset preparation
* **OCR & Preprocessing** – Text extraction from PDFs and scanned documents
* **BERT Model** – Fine-tuning and evaluation
* **LLM Routing Agent** – Category prediction using an LLM
* **RAG Routing Agent** – Category prediction using retrieval-augmented generation
* **Workflow Orchestrator** – Routes documents to the appropriate service agent
* **Service Agents** – Provide guidance for USCIS, DMV, SSA, and IRS documents

> **Note:** This project is developed for educational and research purposes as part of the University of San Diego M.S. in Applied Artificial Intelligence Capstone Project.
