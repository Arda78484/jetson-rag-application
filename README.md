# RAG Application for NVIDIA Jetson Modules

## Overview
This program provides a streamlined testing environment for Retrieval-Augmented Generation (RAG) on NVIDIA Jetson modules. It facilitates easy experimentation with different RAG configurations and local LLM models. You can choose the embedding and LLM model you want to test. The program flowchart is as follows:

![Program Flowchart](./media/flowchart.jpg)

## How to Get an NIM API Key (Free)
1. Go to [NVIDIA Build](https://build.nvidia.com/explore/discover).
2. Create a new account from the top right of the page.
3. Apply for 5000 API credits using your student or company email.

## How to Start the Program
To start the program, run the following command:
```bash
python3 webui.py
```

## Dependencies
Make sure you have the following dependencies installed:
- `gradio`
- `openai`
- `pymilvus`
- `requests`
- `tqdm`
- `typing_extensions`