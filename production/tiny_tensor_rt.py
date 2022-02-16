import ctypes
import time
from multiprocessing.connection import Listener, Client
import numpy as np

import pycuda.autoinit  # IMPORTANT leave this import here
import pycuda.driver as cuda
import threading

import tqdm
from polygraphy.common import TensorMetadata


import tensorrt as trt
from polygraphy.comparator import DataLoader


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTInference:
    def __init__(self, trt_engine_path):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # Load (or create) engine
        if trt_engine_path.endswith('engine'):
            with open(trt_engine_path, 'rb') as f:
                buf = f.read()
                engine = runtime.deserialize_cuda_engine(buf)
        elif trt_engine_path.endswith('onnx'):
            print("Building engine and saving it as 'out.trt' for next loadings...")
            engine = self.build_engine_onnx(trt_engine_path, TRT_LOGGER)
            with open('pcr.engine', 'wb') as f:
                f.write(engine.serialize())
        else:
            raise NameError("The provided file is not an onnx or an engine")
        context = engine.create_execution_context()

        # prepare buffer
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 256 x 256 x 3 ( x 1 )
            # binding = "image"; size = (256, 256, 3)
            # binding = "prediction"; size = (1, 8, 8, 1024)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # dtype = np.float32
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)  # (256 x 256 x 3 ) x (32 / 4)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings

    @staticmethod
    def build_engine_onnx(model_file, trt_logger):
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1)  # EXPLICIT_BATH is 1
        config = builder.create_builder_config()

        config.clear_flag(trt.BuilderFlag.TF32)

        # config.set_flag(trt.BuilderFlag.INT8)
        # config.int8_calibrator = Int8_calibrator

        config.max_workspace_size = 3 * 1 << 30  # Number of byte in a GigaByte
        parser = trt.OnnxParser(network, trt_logger)

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        return builder.build_engine(network, config)

    def infer(self, port):
        li = Listener(('localhost', port), authkey=b'secret password').accept()
        cl = Client(('localhost', port + 1), authkey=b'secret password')

        threading.Thread.__init__(self)
        self.cfx.push()

        while True:
            while not li.poll():  # TODO USE A BETTER DATA STRUCTURE
                continue
            pc = li.recv()

            pc = pc.astype(trt.nptype(trt.float32)).ravel()
            np.copyto(self.inputs[0].host, pc)

            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            # Run inference.
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            # Synchronize the stream
            self.stream.synchronize()

            res = [out.host for out in self.outputs]

            cl.send(res)

        self.cfx.pop()

    def destroy(self):
        self.cfx.pop()


class myThread(threading.Thread):
    def __init__(self, func, port):
        threading.Thread.__init__(self)
        self.func = func
        self.port = port

    def run(self):
        print("Starting thread...")  # + self.args[0])
        self.func(self.port)
        print("Exiting thread")  # + self.args[0])


class BackBone:
    def __init__(self):
        yolo_port = 6000
        it = 10000

        engine = TRTInference('pcr.engine')

        yolo_thread = myThread(engine.infer, yolo_port)
        yolo_thread.start()

        self.yolo_client = Client(('localhost', yolo_port), authkey=b'secret password')
        self.yolo_listener = Listener(('localhost', yolo_port + 1), authkey=b'secret password').accept()

    def __call__(self, x):
        self.yolo_client.send(x)

        while not self.yolo_listener.poll():
            continue
        res = self.yolo_listener.recv()
        _, *weights = res
        import torch
        weights = [torch.tensor(w).cuda().unsqueeze(0) for w in weights]
        res = [[weights[i], weights[i + 1], weights[i + 2]] for i in range(0, 12, 3)]
        return res

if __name__ == '__main__':
    yolo_port = 6000
    it = 100
    #
    engine = TRTInference('pcr.onnx')
    #
    yolo_thread = myThread(engine.infer, yolo_port)
    yolo_thread.start()

    yolo_client = Client(('localhost', yolo_port), authkey=b'secret password')
    yolo_listener = Listener(('localhost', yolo_port + 1), authkey=b'secret password').accept()

    data_loader = DataLoader(iterations=it,
                             val_range=(-0.5, 0.5),
                             input_metadata=TensorMetadata.from_feed_dict(
                                 {'input': np.zeros([1, 2024, 3], dtype=np.float32)}))

    start = time.time()
    for x in tqdm.tqdm(data_loader):
        yolo_client.send(x['input'])

        while not yolo_listener.poll():
            continue
        res = yolo_listener.recv()
    # import onnxruntime as ort
    #
    # backbone = ort.InferenceSession('pcr.onnx')
    #
    # start = time.time()
    # for x in tqdm.tqdm(data_loader):
    #     outputs = backbone.run(None, x)
    #
    # print((time.time() - start) / it)
    # print(1 / ((time.time() - start) / it))
