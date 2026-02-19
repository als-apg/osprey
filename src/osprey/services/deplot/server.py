"""FastAPI application for the DePlot graph extraction service.

Exposes a ``POST /extract`` endpoint that accepts chart images and returns
extracted data tables, and a ``GET /health`` endpoint for service discovery.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("osprey.services.deplot")


def create_app() -> FastAPI:
    """Create and configure the DePlot FastAPI application."""
    app = FastAPI(
        title="OSPREY DePlot Service",
        description="Graph data extraction via Google's DePlot (Pix2Struct) model",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint for service discovery."""
        return {"status": "ok", "service": "deplot"}

    @app.post("/extract")
    async def extract(
        image: UploadFile = File(..., description="Chart image (PNG, JPEG)"),
        preprocess: bool = Query(True, description="Apply OpenCV preprocessing"),
    ) -> JSONResponse:
        """Extract data table from a chart image.

        Args:
            image: Uploaded chart image file.
            preprocess: Whether to apply chart region detection and contrast
                enhancement before extraction.

        Returns:
            JSON with columns, data, raw_table, and title fields.
        """
        image_bytes = await image.read()

        if preprocess:
            import cv2
            import numpy as np

            from osprey.services.deplot.preprocessing import preprocess_chart

            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not decode image"},
                )

            processed = preprocess_chart(img)
            _, buffer = cv2.imencode(".png", processed)
            image_bytes = buffer.tobytes()

        from osprey.services.deplot.model import extract_table, parse_table

        raw = extract_table(image_bytes)
        result = parse_table(raw)

        return JSONResponse(content=result)

    return app
