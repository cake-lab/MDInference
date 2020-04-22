#!env python

import tensorflow as tf

from mdinference.models import ServingModel

class TRTServingModel(ServingModel):
    tf = __import__('tensorflow')
    trt = __import__('tensorrt') #import tensorrt as trt
    def __init__(self, model_file, precision_mode, input_layer, output_layer, input_height, input_width=None, accuracy=0.5, window_size=100, **kwargs):
        super(TFServingModel, self).__init__(model_file, input_layer, output_layer, input_height, input_width, accuracy, window_size, **kwargs)
        self.precision_mode = precision_mode
        self.sess, self.input_operation, self.output_operation = self.getGraphData(model_file, input_layer, output_layer, precision_mode)
        self.model_name += "." + self.precision_mode
    
    def __str__(self):
        return "(%s, %s, %s, %s)" % (os.path.basename(self.model_file), self.mu, self.sigma, self.precision_mode)
        
    @staticmethod
    def loadBaseline(model_file): #getResnet50():
        with gfile.FastGFile("resnetV150_frozen.pb",'rb') as f:
            graph_def = TRTServingModel.tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    
    @staticmethod
    def getPrecision(in_graph_def, precision_mode, input_name, batch_size=128,workspace_size=1<<30):
        if precision_mode.lower() == "native":
            return in_graph_def
            
        trt_graph = TRTServingModel.trt.create_inference_graph(in_graph_def,
                                                    input_name,
                                                    max_batch_size=batch_size,
                                                    max_workspace_size_bytes=workspace_size,
                                                    precision_mode=precision_mode)  # Get optimized graph
        return trt_graph
    
    @staticmethod
    def getFP32(batch_size=128,workspace_size=1<<30):
        return getPrecision("FP32")
        
    @staticmethod
    def getFP16(batch_size=128,workspace_size=1<<30):
        return getPrecision("FP16")
    
    @staticmethod
    def getINT8(batch_size=128,workspace_size=1<<30):
        calibGraph = getPrecision("INT8")
        # Then we're supposed to run some calibration data... but that'll be done later?
        # timings,comp,_,mdstats=timeGraph(calibGraph,f.batch_size,1,dummy_input)
        return rt.calib_graph_to_infer_graph(calibGraph)
    
    
    
    
    @staticmethod
    def getGraphData(model_file, input_layer, output_layer, precision_mode="native"):
        
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        
        graph = TFServingModel.tf.Graph()
        graph_def = loadBaseline(model_file)
        precision_graph_def = getPrecision(graph_def, precision_mode, input_name)
        
        with graph.as_default():
            TFServingModel.tf.import_graph_def(precision_graph_def)
            
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        sess = TFServingModel.tf.Session(graph=graph, gpu_options=gpu_options)
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);
        
        return (sess, input_operation, output_operation)

    #Override method
    def runInference(self, input_t):
        logging.info("TF Inference")
        return self.sess.run(self.output_operation.outputs[0],{self.input_operation.outputs[0]: input_t})
    
    
    
    
