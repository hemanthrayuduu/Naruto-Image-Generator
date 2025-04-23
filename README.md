# Naruto Text-to-Image Generator

This project is a full-stack web application that generates Naruto-style images based on user text prompts using a fine-tuned Stable Diffusion model.

## Features

-   **Text-to-Image Generation:** Uses Stable Diffusion with a LoRA adapter fine-tuned on Naruto images.
-   **Streamlit Frontend:** Interactive web interface for entering prompts and viewing results.
-   **FastAPI Backend:** Handles image generation requests and serves the model.
-   **Advanced Settings:** Control inference steps, guidance scale, image dimensions, and seed.
-   **Recent Generations Gallery:** Displays the last 4 generated images (session-based).
-   **Downloadable Images:** Download generated images with descriptive filenames.
-   **Themed UI:** Basic Naruto-inspired styling for status messages.

## Project Structure

```
├── backend/
│   ├── app.py             # FastAPI application
│   ├── pipeline.py        # Stable Diffusion pipeline logic
│   ├── requirements.txt   # Backend Python dependencies
│   └── model/             # Directory for the fine-tuned LoRA model
│       ├── adapter_config.json
│       └── adapter_model.safetensors.gz (Compressed model file)
├── frontend/
│   ├── app.py             # Streamlit application
│   ├── style.css          # Custom CSS for frontend
│   └── requirements.txt   # Frontend Python dependencies
├── .gitignore             # Files/directories ignored by Git
└── README.md              # This file
```

## Setup

**Prerequisites:**

*   Python 3.9+ 
*   `pip` and `venv` (standard Python modules)
*   Git

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/hemanthrayuduu/Naruto-Image-Generator.git
    cd Naruto-Image-Generator
    ```

2.  **Uncompress the Model File:**
    ```bash
    # Navigate to the model directory
    cd backend/model
    # Uncompress the model file
    gzip -d adapter_model.safetensors.gz
    # Go back to the project root
    cd ../..
    ```

3.  **Setup Backend Environment:**
    ```bash
    cd backend
    python -m venv .venv
    source .venv/bin/activate  # Or relevant activation command for your OS
    python -m pip install -r requirements.txt
    # Ensure model files are in backend/model/
    cd .. 
    ```

4.  **Setup Frontend Environment:**
    ```bash
    cd frontend
    python -m venv .venv
    source .venv/bin/activate  # Or relevant activation command for your OS
    pip install -r requirements.txt
    cd ..
    ```

## Running the Application

1.  **Start the Backend Server:**
    *   Open a terminal.
    *   Navigate to the `backend` directory: `cd backend`
    *   Activate the backend virtual environment: `source .venv/bin/activate`
    *   Run Uvicorn: 
        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```
    *   Keep this terminal running.

2.  **Start the Frontend Application:**
    *   Open a **new** terminal.
    *   Navigate to the **project root** directory (e.g., `Naruto-Image-Generator`).
    *   Activate the frontend virtual environment: `source frontend/.venv/bin/activate`
    *   Run Streamlit:
        ```bash
        streamlit run frontend/app.py
        ```

3.  **Access the App:** Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## Configuration

*   **Frontend:** The backend URL is currently hardcoded in `frontend/app.py`. For deployment, consider using Streamlit Secrets or environment variables.
*   **Backend:** Expects the LoRA model files (`adapter_config.json`, `adapter_model.safetensors`) to be present in the `backend/model/` directory. 

## Note on the Model File

The adapter model file (`adapter_model.safetensors`) is distributed in a compressed format (`.gz`) to reduce repository size. Make sure to uncompress it before running the application as described in the setup instructions. 