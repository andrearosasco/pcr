import time

import torch
from polygraphy.backend.onnx import OnnxFromPath, GsFromOnnx
from polygraphy.backend.onnxrt import SessionFromOnnx, OnnxrtRunner
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader, Comparator
import numpy as np
import onnxruntime as ort

if __name__ == '__main__':

    onnx_file = 'assets/production/pcr.onnx'
    #
    # data_loader = DataLoader(iterations=100,
    #                          val_range=(-0.5, 0.5),
    #                          input_metadata=TensorMetadata.from_feed_dict({'input': np.zeros([1, 2024, 3], dtype=np.float32)}))
    #
    # for x in data_loader:
    #     print(x)

    # build_onnx = SessionFromOnnx(onnx_file)

    # ort_sess = ort.InferenceSession('assets/pcr.onnx')

    # for i, x in enumerate(data_loader):
    #     if i == 1:
    #         start = time.time()
    #     outputs = ort_sess.run(None, x)
    # print(i)
    # print(time.time() - start)
    #
    # with OnnxrtRunner(build_onnx) as runner:
    #     start = time.time()
    #     for i, x in enumerate(data_loader):
    #         outputs = runner.infer(feed_dict=x)

    # print(i)
    # print(time.time() - start)

    # runners = [OnnxrtRunner(build_onnx)]
    # run_results = Comparator.run(runners, data_loader=data_loader)

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file))
    engine = build_engine()

    with open('assets/production/pcr_polygraphy.engine', 'wb') as f:
        f.write(engine.serialize())
    # print(Comparator.compare_accuracy(run_results))