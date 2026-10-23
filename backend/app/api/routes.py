"""
FastAPI route definitions for health checks and image classification (K-Means & DBSCAN).
"""
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from backend.app.schemas.classification import HealthResponse, ClassificationResponse
from backend.app.services.classifier import LandClassifierService
from backend.app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Service health check endpoint.
    """
    return HealthResponse(
        status="ok",
        app=settings.PROJECT_NAME,
        version=settings.VERSION,
    )


@router.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_satellite_image(
    file: UploadFile = File(..., description="Satellite or aerial image file (PNG, JPG, TIF)"),
    algorithm: str = Query(
        default="kmeans",
        pattern="^(kmeans|dbscan)$",
        description="Clustering algorithm to use: 'kmeans' or 'dbscan'",
    ),
    n_clusters: int = Query(
        default=5,
        ge=2,
        le=10,
        description="Number of K-Means clusters (used if algorithm='kmeans')",
    ),
    dbscan_eps: float = Query(
        default=5.0,
        ge=0.5,
        le=50.0,
        description="Epsilon neighborhood radius in CIELAB (used if algorithm='dbscan')",
    ),
    dbscan_min_samples: int = Query(
        default=25,
        ge=2,
        le=500,
        description="Minimum core samples in neighborhood (used if algorithm='dbscan')",
    ),
    apply_blur: bool = Query(
        default=True,
        description="Whether to apply Gaussian blur to suppress sensor noise",
    ),
    blur_kernel: int = Query(
        default=3,
        ge=1,
        description="Gaussian blur kernel size (positive odd integer)",
    ),
    refine_mask: bool = Query(
        default=True,
        description="Whether to apply morphological refinement (opening & closing)",
    ),
    refine_kernel: int = Query(
        default=3,
        ge=1,
        description="Morphological structuring element size (positive odd integer)",
    ),
):
    """
    Process an uploaded satellite image through K-Means or DBSCAN clustering,
    semantic land-cover labeling, and morphological refinement.
    """
    if blur_kernel % 2 == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"blur_kernel must be an odd integer, got {blur_kernel}",
        )
    if refine_kernel % 2 == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"refine_kernel must be an odd integer, got {refine_kernel}",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum size limit of {settings.MAX_FILE_SIZE_MB}MB.",
            )

        result = LandClassifierService.classify(
            image_bytes=content,
            algorithm=algorithm,
            n_clusters=n_clusters,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            apply_blur=apply_blur,
            blur_kernel=blur_kernel,
            refine=refine_mask,
            refine_kernel=refine_kernel,
        )
        return result

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during classification: {str(exc)}",
        )
