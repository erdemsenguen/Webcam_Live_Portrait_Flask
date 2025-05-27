import signal
import sys
from inference import Inference
from api_server import APIServer
import os

path = os.path.dirname(os.path.abspath(__file__))
inference = Inference()
inference.set_parameters(
    **{"roll": 1, "yaw": 1, "pitch": 1, "expression": 1, "scale": 1, "t": 1}
)
api_server = APIServer(
    set_source_funct=inference.set_source,
    stop_funct=inference.stop,
    status_funct=inference.status_funct,
    run_funct=inference.set_run,
    set_param=inference.set_parameters,
    set_greenscreen=inference.set_greenscreen,
    source_img_dir=f"{os.path.dirname(path)}/photos",
    source_green_dir=f"{os.path.dirname(path)}/Backgrounds",
)


def shutdown_handler(sig, frame):
    api_server.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
try:
    api_server.start()
    inference.main()
except Exception as e:
    sys.exit(1)
finally:
    api_server.kill()
