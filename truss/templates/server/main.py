import os

from truss_server import TrussServer

CONFIG_FILE = "config.yaml"

if __name__ == "__main__":
    if os.environ.get("TRUSS_DIAG_START_METHOD"):  # fork-only diagnostic
        import multiprocessing

        multiprocessing.set_start_method(os.environ["TRUSS_DIAG_START_METHOD"])
    http_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    TrussServer(http_port, CONFIG_FILE).start()
