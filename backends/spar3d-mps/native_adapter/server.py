"""LocalAI 4.11 external gRPC backend for the native PyTorch/MPS SPAR3D model."""
import argparse
from concurrent import futures
import json
import logging
import os
import re
from pathlib import Path
import signal
import threading
import time

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import grpc
import backend_pb2 as pb
import backend_pb2_grpc as rpc

LOG = logging.getLogger("spar3d")



class Backend(rpc.BackendServicer):
    def __init__(self):
        self.engine = None
        self.busy = False
        self.lock = threading.Lock()

    def Health(self, request, context):
        return pb.Reply(message=b"OK")

    def Status(self, request, context):
        state = pb.StatusResponse.BUSY if self.busy else (
            pb.StatusResponse.READY if self.engine else pb.StatusResponse.UNINITIALIZED)
        return pb.StatusResponse(state=state)

    def ModelMetadata(self, request, context):
        return pb.ModelMetadataResponse()

    def LoadModel(self, request, context):
        with self.lock:
            self.busy = True
            try:
                if not request.ModelFile:
                    raise ValueError("LocalAI ModelFile path is required")
                root = Path(request.ModelFile).resolve(strict=True)
                if self.engine:
                    if self.engine.root == root:
                        return pb.Result(success=True, message="Model already loaded")
                    raise ValueError("Unload the current model before loading another")
                from engine import Engine
                start = time.monotonic()
                self.engine = Engine(root)
                LOG.info("Loaded model in %.2fs", time.monotonic() - start)
                return pb.Result(success=True, message="SPAR3D PyTorch/MPS loaded")
            except Exception as error:
                LOG.exception("Model loading failed")
                return pb.Result(success=False, message=str(error))
            finally:
                self.busy = False

    def Free(self, request, context):
        with self.lock:
            if self.engine:
                self.engine.close()
                self.engine = None
            return pb.Result(success=True, message="Model released")

    def Generate3D(self, request, context):
        with self.lock:
            self.busy = True
            try:
                if self.engine is None:
                    raise RuntimeError("SPAR3D model is not loaded")
                if not request.src or not Path(request.src).is_file():
                    raise ValueError("Conditioning image path must point to an existing file")
                if not request.dst:
                    raise ValueError("GLB destination path is required")
                quality = request.quality or "auto"  # Protocol defines empty as auto.
                if quality != "auto":
                    raise ValueError("SPAR3D quality must be auto.")
                unknown = set(request.params) - {"target_triangles"}
                if unknown:
                    raise ValueError("Unsupported SPAR3D parameters: " + ", ".join(sorted(unknown)))
                shape_steps = request.step if request.step != 0 else 10
                if shape_steps < 1:
                    raise ValueError("Shape steps must be positive; triangle target = 1000 × steps")
                target_triangles = shape_steps * 1000
                if "target_triangles" in request.params:
                    value = request.params["target_triangles"]
                    if not re.fullmatch(r"[0-9]+", value) or int(value) < 1:
                        raise ValueError("target_triangles must be a positive integer string")
                    target_triangles = int(value)
                material_steps = request.texture_steps if request.texture_steps != 0 else 32
                if material_steps < 1:
                    raise ValueError("Material steps must be positive; texture size = 32 × steps")
                if request.cfg_scale != 0 or request.seed != 0:
                    raise ValueError("SPAR3D does not support Guidance or Seed; leave them blank")
                result = self.engine.generate(request.src, request.dst, 32 * material_steps, request.background or "auto", target_triangles=target_triangles,
                                              allow_larger_target="target_triangles" not in request.params)
                return pb.Result(success=True, message="GLB generated", metadata=json.dumps(result).encode())
            except (ValueError, FileNotFoundError) as error:
                LOG.error("Invalid generation request: %s", error)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            except Exception as error:
                LOG.exception("Generation failed")
                context.abort(grpc.StatusCode.INTERNAL, str(error))
            finally:
                self.busy = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", "-addr", required=True)
    args = parser.parse_args()
    log = Path(__file__).resolve().parent.parent / "runtime" / "native-backend.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log)])
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    rpc.add_BackendServicer_to_server(Backend(), server)
    if not server.add_insecure_port(args.addr):
        raise RuntimeError(f"Could not bind gRPC address {args.addr}")
    server.start()
    LOG.info("LocalAI SPAR3D PyTorch/MPS backend listening on %s", args.addr)
    def shutdown(signum, frame):
        server.stop(0)
        os._exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
