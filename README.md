uvicorn demo.main:app --reload --reload-dir src
uvicorn demo.main:app --reload --reload-exclude "logs/_" --reload-exclude "data/_" --reload-exclude "\*.log"
