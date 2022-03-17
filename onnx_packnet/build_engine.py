import argparse
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import common
import logging


TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EngineBuilder")
logger.setLevel(logging.INFO)


def build_engine(onnx_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print ('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print (parser.get_error(error))
            return None
    return builder.build_engine(network, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="ONNX model file")
    parser.add_argument("--engine", required=True, type=str, help="The file to save engine to")
    args = parser.parse_args()
    engine = build_engine(args.model)

    logger.info("Serializing engine to file: {:}".format(args.engine))
    with open(args.engine, "wb") as f:
        f.write(engine.serialize())


