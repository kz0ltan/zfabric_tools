class RetrievalError(Exception):
    pass


class FailedRetrievalError(Exception):
    def __init__(self, msg, retrievals):
        self.message = msg
        self.retrievals = retrievals
        super().__init__(self.message)


class ExtractionError(Exception):
    pass


class HTTPStatusError(Exception):
    def __init__(self, response):
        self.response = response
        self.status_code = response.status
        self.message = f"HTTP {response.status}: {response.status_text}"
        super().__init__(self.message)
