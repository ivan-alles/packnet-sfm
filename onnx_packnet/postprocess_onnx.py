import torch
import onnx
import numpy as np
import argparse
import onnx_graphsurgeon as gs
from post_processing import *
import sys
import os


def post_process_model(input_file, output_file, opset=11):
    """
    Use ONNX graph surgeon to replace upsample and instance normalization nodes. Refer to post_processing.py for details.
    Args:
        input_file : Path to ONNX file
    """
    # Load the packnet graph
    graph = gs.import_onnx(onnx.load(input_file))

    if opset>=11:
        graph = process_pad_nodes(graph)

    # Replace the subgraph of upsample with a single node with input and scale factor.
    if torch.__version__ < '1.5.0':
        graph = process_upsample_nodes(graph, opset)

    # Convert the group normalization subgraph into a single plugin node.
    graph = process_groupnorm_nodes(graph)

    # Remove unused nodes, and topologically sort the graph.
    graph.cleanup().toposort()

    print(f'Saving the ONNX model to {input_file}')

    # Export the onnx graph from graphsurgeon
    onnx.save_model(gs.export_onnx(graph), output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Post-process ONNX model")
    parser.add_argument("--input", help="Path to input ONNX model", required=True, type=str)
    parser.add_argument("--output", help="Path to output ONNX model", required=True, type=str)
    parser.add_argument("--opset", type=int, help="ONNX opset to use", default=11)

    args = parser.parse_args()
    onnx.checker.check_model(args.input)

    post_process_model(args.input, args.output, args.opset)
