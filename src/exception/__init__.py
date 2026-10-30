import sys
import logging
from typing import Union, Any


def error_message_detail(error: Union[Exception, str], error_detail: Any = None) -> str:
    """
    Extract detailed error info: file, line number, message.
    Safely handles both Exception instances and string messages,
    as well as cases where exc_info traceback is None.
    """
    exc_type, exc_val, exc_tb = error_detail.exc_info() if error_detail else (None, None, None)

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in script: [{file_name}] "
            f"at line [{line_number}] : [{str(error)}]"
        )
    else:
        error_message = f"Error occurred: [{str(error)}]"

    logging.error(error_message)
    return error_message


class MyException(Exception):
    """
    Custom project-wide exception class.
    Accepts either an Exception object or a string error message.
    """

    def __init__(self, error: Union[Exception, str], error_detail: Any = None):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail or sys)

    def __str__(self):
        return self.error_message
