import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import html
import logging
import json

class APIServer:
    def __init__(self,set_source_funct:callable,stop_funct:callable,status_funct:callable,run_funct:callable, set_param:callable,host="127.0.0.1", port=5001, source_img_dir:str=None,):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)
        self._register_routes()
        self._thread=None
        self.set_param=set_param
        self.source_img_dir=source_img_dir
        self.extensions = ('.jpg')
        self.set_source_funct=set_source_funct
        self.stop_funct=stop_funct
        self.status_funct=status_funct
        self.run_funct=run_funct
        # Get files and remove extensions
        self.file_names = [
                        os.path.splitext(f)[0]
                        for f in os.listdir(self.source_img_dir)
                        if f.lower().endswith(self.extensions)
                    ]
    def _register_routes(self):
        @self.app.route("/api/data", methods=["POST","GET"])
        def handle_request():
            import platform
            if platform.system() == "Windows":
                import pythoncom
                pythoncom.CoInitialize()
            if request.method == "POST":
                json_data = request.get_json()
                j_input = html.escape(json_data.get("input"))
                if j_input:
                    if self.source_img_dir==None:
                        return jsonify({
                        "error": "Server initialization error, Image directory does not exist."
                    }), 401
                    else:
                        inp=str(j_input)
                        if inp in self.file_names:
                            try:
                                if int(j_input)==7 or int(j_input)==11:
                                    self.set_param(**{"pitch":1,
                                        "yaw":1,
                                        "roll":1,
                                        "expression":1.05,
                                        "scale":1.1,
                                        "t":1})
                                else:
                                    self.set_param(**{"roll":0.7,"yaw":0.7,"pitch":0.7,"expression":1.05,"scale":1.1,"t":1})
                                self.set_source_funct(f"{self.source_img_dir}/{j_input}.jpg")
                                self.handle_increment(int(j_input))
                                return jsonify({"status": "success", "input": j_input}), 200
                            except Exception as e:
                                print(e)
                                return jsonify({"error": f"Software exception occured \n{e}"}),404
                        else:
                            return jsonify({"error": f"Specified file does not exist."})
                if j_input is None:
                    return jsonify({
                        "error": "Data type is not supported. Send a JSON with 'input' key."
                    }), 400
            elif request.method =="GET":
                return jsonify({"status": self.status_funct()}),200
        @self.app.route("/api/run", methods=["GET"])
        def handle_run_request():
            self.run_funct()
            return jsonify({"status":"Running"}),200
    def handle_increment(self,photo_id:int):
            data_file = "data.json"
            if not os.path.exists(data_file):
                with open(data_file, "w") as f:
                    json.dump({"1":["Angela Merkel",0],
                               "2":["Olaf Scholz",0],
                               "3":["Chris Hemsworth",0],
                               "4":["Heidi Klum",0],
                               "5":["Scarlet Johannson",0],
                               "6":["Cristoph Waltz",0],
                               "7":["Mona Lisa",0],
                               "8":["Matthias Schweighöfer",0],
                               "9":["Jenna Ortega",0],
                               "10":["Henry Cavill",0],
                               "11":["Albrecht Dürer",0]}, f)   

            else:
                with open(data_file, "r") as f:
                    data = json.load(f)

                photo_id_str = str(photo_id)
                if photo_id_str in data:
                    data[photo_id_str][1] += 1  # Increment the count

                with open(data_file, "w") as f:
                    json.dump(data, f, indent=2)
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