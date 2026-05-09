from fastapi import HTTPException, Request, status

from services.inference_runtime import InferenceRuntime

def get_inference_runtime(request: Request) -> InferenceRuntime:
    runtime = getattr(request.app.state, "inference_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference runtime not available.",
        )
    return runtime