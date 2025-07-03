import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import html
import logging
import json


class APIServer:
    def __init__(
        self,
        set_source_funct: callable,
        stop_funct: callable,
        status_funct: callable,
        run_funct: callable,
        active_img_funct:callable,
        set_param: callable,
        set_greenscreen: callable,
        host="0.0.0.0",
        port=5001,
        source_img_dir: str = None,
        source_green_dir: str = None,
    ):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        self._register_routes()
        self._thread = None
        self.set_param = set_param
        self.source_img_dir = source_img_dir
        self.source_green_dir = source_green_dir
        self.extensions = ".jpg"
        self.set_source_funct = set_source_funct
        self.stop_funct = stop_funct
        self.status_funct = status_funct
        self.run_funct = run_funct
        self.active_img_funct=active_img_funct
        self.green_funct = set_greenscreen
        self.file_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.source_img_dir)
            if f.lower().endswith(self.extensions)
        ]
        self.green_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.source_green_dir)
            if f.lower().startswith("meeting")
        ]

    def _register_routes(self):
        @self.app.route("/api/data", methods=["POST", "GET"])
        def handle_request():
            import platform

            if platform.system() == "Windows":
                import pythoncom

                pythoncom.CoInitialize()
            if request.method == "POST":
                json_data = request.get_json()
                j_input = html.escape(json_data.get("input"))
                if j_input:
                    if self.source_img_dir == None:
                        return (
                            jsonify(
                                {
                                    "error": "Server initialization error, Image directory does not exist."
                                }
                            ),
                            401,
                        )
                    else:
                        inp = str(j_input)
                        if inp in self.file_names:
                            try:
                                self.set_source_funct(
                                    f"{self.source_img_dir}/{j_input}.jpg"
                                )
                                return (
                                    jsonify({"status": "success", "input": j_input}),
                                    200,
                                )
                            except Exception as e:
                                print(e)
                                return (
                                    jsonify(
                                        {"error": f"Software exception occured \n{e}"}
                                    ),
                                    404,
                                )
                        else:
                            return jsonify({"error": f"Specified file does not exist."})
                if j_input is None:
                    return (
                        jsonify(
                            {
                                "error": "Data type is not supported. Send a JSON with 'input' key."
                            }
                        ),
                        400,
                    )
            elif request.method == "GET":
                if self.status_funct():
                    return jsonify({"status": self.active_img_funct()}), 200
                else:
                    return jsonify({"status": False}), 200
        @self.app.route("/api/run", methods=["GET"])
        def handle_run_request():
            self.run_funct()
            return jsonify({"status": "Running"}), 200
        @self.app.route("/api/green", methods=["POST"])
        def handle_green_screen_request():
            json_data = request.get_json()
            j_input = html.escape(json_data.get("input"))
            if j_input:
                if self.source_img_dir == None:
                    return (
                        jsonify(
                            {
                                "error": "Server initialization error, Image directory does not exist."
                            }
                        ),
                        401,
                    )
                else:
                    inp = str(j_input)
                    if inp in self.green_names:
                        try:
                            self.green_funct(f"{self.source_green_dir}/{inp}.jpg")
                            return jsonify({"status": "success", "input": j_input})
                        except Exception as e:
                            print(e)
                            return (
                                jsonify({"error": f"Software exception occured \n{e}"}),
                                404,
                            )
            else:
                return jsonify({"error": "WTF"})

    def start(self):
        self._thread = threading.Thread(
            target=lambda: self.app.run(
                host=self.host, port=self.port, debug=False, use_reloader=False
            ),
            daemon=True,
        )
        self._thread.start()

    def kill(self):
        self.stop_funct()
        self._thread.join(timeout=1)
