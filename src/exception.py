import sys
from src.logger import logging
  

def error_message_detail(error):
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_message


class CustomException(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error_message = error_message_detail(error)

    def __str__(self):
        return self.error_message

    # def display_error_message(self):
    #     print(self.error_message)


# Example usage
def divide(a, b):
    try:
        return a / b
    except Exception as e:
        raise CustomException(e)


if __name__ == "__main__":
    try:
        result = divide(10, 0)
        print(result)
    except CustomException as e:
        logging.info("Devide by zero")
        print(e)
