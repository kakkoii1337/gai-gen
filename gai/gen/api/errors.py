from fastapi.responses import JSONResponse

class ErrorResponse(JSONResponse):
    def __init__(self, status_code, code, message):
        super().__init__(
            content={
                "type": "error",
                "code": code,
                "message": message
            },
            status_code=status_code
        )

class ContextLengthExceededError(ErrorResponse):
    def __init__(self):
        super().__init__(
            status_code=400,
            code="context_length_exceeded",
            message="The message has exceeded the model's context length."
        )

class ModelServiceMismatchError(ErrorResponse):
    def __init__(self):
        super().__init__(
            status_code=400,
            code="model_service_mismatch",
            message="The model does not correspond to this service."
        )

class InternalError(ErrorResponse):
    def __init__(self, message):
        super().__init__(
            status_code=500,
            code="internal_error",
            message=message
        )
