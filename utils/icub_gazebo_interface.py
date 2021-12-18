import numpy as np
import yarp


class iCubGazebo:

    def __init__(self, rgb_port="/icubSim/cam/left/rgbImage:o", depth_port='/icubSim/cam/left/depthImage:o'):
        yarp.Network.init()

        # Create a port and connect it to the iCub simulator virtual camera
        self.rgb_port, self.depth_port = yarp.Port(), yarp.Port()
        self.rgb_port.open("/rgb-port")
        self.depth_port.open("/depth-port")
        yarp.Network.connect(rgb_port, "/rgb-port")
        yarp.Network.connect(depth_port, "/depth-port")

        self.rgb_array = np.zeros((240, 320, 3), dtype=np.uint8)
        self.rgb_image = yarp.ImageRgb()
        self.rgb_image.resize(320, 240)
        self.rgb_image.setExternal(self.rgb_array, self.rgb_array.shape[1], self.rgb_array.shape[0])

        self.depth_array = np.zeros((240, 320), dtype=np.float32)
        self.depth_image = yarp.ImageFloat()
        self.depth_image.resize(320, 240)
        self.depth_image.setExternal(self.depth_array, self.depth_array.shape[1], self.depth_array.shape[0])

    def read(self):
        self.rgb_port.read(self.rgb_image)
        self.depth_port.read(self.depth_image)

        return self.rgb_array[..., ::-1], self.depth_array
