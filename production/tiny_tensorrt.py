from multiprocessing.connection import Listener, Client
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # IMPORTANT leave this import here
import pycuda.driver as cuda
import threading
from utils.misc import HostDeviceMem, myThread


class TRTInference:
    def __init__(self, trt_engine_path, trt_engine_datatype, batch_size):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime = trt.Runtime(TRT_LOGGER)

        # Load (or create) engine
        if trt_engine_path.endswith('trt'):
            with open(trt_engine_path, 'rb') as f:
                buf = f.read()
                engine = runtime.deserialize_cuda_engine(buf)
        elif trt_engine_path.endswith('onnx'):
            print("Building engine and saving it as 'out.trt' for next loadings...")
            engine = self.build_engine_onnx(trt_engine_path, TRT_LOGGER)
            with open('out.trt', 'wb') as f:
                f.write(engine.serialize())
        else:
            raise NameError("The provided file is not an onnx or a trt")
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
        config.max_workspace_size = 7 * 1 << 30  # Number of byte in a GigaByte
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
            img = li.recv()

            np.copyto(self.inputs[0].host, img)

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
#
# if __name__ == '__main__':
#     yolo_port = 6000
#     bbone_port = 6005
#     heads_port = 6010
#     max_batch_size = 1
#
#     # Load yolo
#     trt_yolo = TRTInference('mods/effnet-l/yolov4_1_3_256_256_static.onnx',
#                             trt_engine_datatype=trt.DataType.FLOAT,
#                             batch_size=max_batch_size)
#     yolo_thread = myThread(trt_yolo.infer, yolo_port)
#     yolo_thread.start()
