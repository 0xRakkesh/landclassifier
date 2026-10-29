"""
FastAPI route definitions for health checks and Supervised Random Forest land classification.
"""
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from backend.app.schemas.classification import HealthResponse, ClassificationResponse
from backend.app.services.classifier import LandClassifierService
from backend.app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check endpoint."""
    return HealthResponse(
        status="ok",
        app=settings.PROJECT_NAME,
        version=settings.VERSION,
    )

@router.get("/metrics", tags=["Evaluation"])
async def get_test_benchmark_metrics():
    """Retrieve official test-set accuracy, precision, recall, and mIoU benchmark metrics."""
    return LandClassifierService.get_benchmark_metrics()

@router.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_satellite_image(
    file: UploadFile = File(..., description="Satellite image tile (PNG, JPG, TIF)"),
    mask_file: Optional[UploadFile] = File(default=None, description="Optional ground truth raster mask"),
    apply_blur: bool = Query(default=True, description="Whether to apply smoothing filter to suppress sensor noise"),
    blur_kernel: int = Query(default=5, ge=1, description="Filter kernel diameter (positive odd integer)"),
    use_bilateral: bool = Query(default=True, description="Whether to use edge-preserving bilateral filtering"),
    refine_mask: bool = Query(default=True, description="Whether to apply morphological refinement"),
    refine_kernel: int = Query(default=3, ge=1, description="Morphological structuring element size"),
):
    """
    Process an uploaded satellite image through the calibrated Supervised Random Forest model.
    Computes accuracy, precision, recall, and IoU when ground-truth mask is provided or auto-detected.
    """
    if blur_kernel % 2 == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"blur_kernel must be odd, got {blur_kernel}")
    if refine_kernel % 2 == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"refine_kernel must be odd, got {refine_kernel}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    mask_bytes = None
    if mask_file is not None:
        mask_bytes = await mask_file.read()

    try:
        result = LandClassifierService.classify(
            image_bytes=image_bytes,
            file_name=file.filename,
            algorithm="random_forest",
            apply_blur=apply_blur,
            blur_kernel=blur_kernel,
            use_bilateral=use_bilateral,
            refine=refine_mask,
            refine_kernel=refine_kernel,
            gt_mask_bytes=mask_bytes,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
