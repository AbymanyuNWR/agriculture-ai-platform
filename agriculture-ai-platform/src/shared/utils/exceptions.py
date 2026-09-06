from typing import Any, Optional

class AgricultureAIException(Exception):
    """Base exception for Agriculture AI Platform"""
    
    def __init__(
        self,
        message: str = "An error occurred",
        status_code: int = 500,
        detail: Optional[Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)

class ValidationError(AgricultureAIException):
    """Validation error"""
    
    def __init__(self, message: str = "Validation error", detail: Optional[Any] = None):
        super().__init__(message=message, status_code=400, detail=detail)

class AuthenticationError(AgricultureAIException):
    """Authentication error"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message=message, status_code=401)

class AuthorizationError(AgricultureAIException):
    """Authorization error"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message=message, status_code=403)

class NotFoundError(AgricultureAIException):
    """Resource not found error"""
    
    def __init__(self, resource: str = "Resource", resource_id: Any = None):
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} with id {resource_id} not found"
        super().__init__(message=message, status_code=404)

class RateLimitError(AgricultureAIException):
    """Rate limit exceeded error"""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message="Rate limit exceeded",
            status_code=429,
            detail={"retry_after": retry_after}
        )

class ModelNotFoundError(AgricultureAIException):
    """ML model not found error"""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model {model_name} not found",
            status_code=404
        )

class PredictionError(AgricultureAIException):
    """Prediction error"""
    
    def __init__(self, message: str = "Prediction failed"):
        super().__init__(message=message, status_code=500)
