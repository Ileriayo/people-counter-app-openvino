#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.ie_plugin = None
        self.net = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None

    def load_model(self, model_xml, device, cpu_extension):
        # Create IECore
        log.info('Creating IECore...')
        self.ie_plugin = IECore()

        # Add CPU extension to IECore
        if cpu_extension and 'CPU' in device:
            log.info('Adding CPU extension:\n\t{}'.format(cpu_extension))
            self.ie_plugin.add_extension(cpu_extension, device)

        # Load the IR model into IENetwork
        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        log.info("Loading model into IENetwork:\n\t{}\n\t{}".format(model_xml, model_bin))
        
        # self.net = IENetwork(model=model_xml, weights=model_bin) # Reading network using constructor is deprecated
        self.net = self.ie_plugin.read_network(model=model_xml, weights=model_bin)

        # Check layers
        log.info('Current device specified: {}'.format(device))
        log.info("Checking for unsupported layers...")
        supported_layers = self.ie_plugin.query_network(network=self.net, device_name='CPU')
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error('These layers are unsupported:\n{}'.format(', '.join(unsupported_layers)))
            log.error('Specify an available extension to add to IECore from the command line using -l/--cpu_extension')
            exit(1)
        else:
            log.info('All layers are supported!')

        # Load the model network into IECore
        self.exec_network = self.ie_plugin.load_network(self.net, device)
        log.info("IR Model has been successfully loaded into IECore")

        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        return self.exec_network

    def get_input_shape(self):
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, image):
        # Async request
        self.infer_request_handle = self.exec_network.start_async(0, { self.input_blob: image })
        return self.infer_request_handle

    def wait(self):
        # Wait for completion of request
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        # Extract and return the output results
        return self.exec_network.requests[0].outputs[self.output_blob]
