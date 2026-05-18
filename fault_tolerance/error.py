from enum import Enum
import os

class ErrorType(Enum):
    NPUNotAvailable = "NPUNotAvailable"
    HCCLTimeOut = "HCCLTimeOut"


class NPUNotAvailable(Exception):
    def __init__(self, error_type: ErrorType, message: str):
        super().__init__(message)
        self.error_type = error_type

class HCCLTimeOut(Exception):
    def __init__(self, error_type: ErrorType, message: str):
        super().__init__(message)
        self.error_type = error_type


def SimulateError(error_type, error_message,error_rank):
    if os.environ['RANK'] == str(error_rank):
        print(f"error_rank:{error_rank},error_type:{error_type}")
        if error_type == ErrorType.NPUNotAvailable:
            raise NPUNotAvailable(error_type, error_message)
        elif error_type == ErrorType.HCCLTimeOut:
            raise HCCLTimeOut(error_type, error_message)
