"""Convenience entry point: `python run.py` starts the API on localhost with auto-reload."""
import uvicorn

import config

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.IS_DEV,
    )
