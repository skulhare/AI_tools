import tensorflow as tf
from tensorflow.python.platform import gfile

# Model file path
model_file = r"output_graph.pb"

# Load the trained model.
with tf.Session() as sess:
    with gfile.FastGFile(model_file,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def,name='')

"""
Example : Names of last layers :
module_apply_default/InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/BatchNorm/Const
module_apply_default/InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/BatchNorm/Const
module_apply_default/InceptionV3/InceptionV3/Mixed_7c/Branch_2/concat/axis
module_apply_default/InceptionV3/InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/BatchNorm/Const
module_apply_default/InceptionV3/InceptionV3/Mixed_7c/concat/axis
module_apply_default/InceptionV3/Logits/GlobalPool/reduction_indices
final_retrain_ops/weights/final_weights
final_retrain_ops/biases/final_biases

"""

with tf.Session() as sess:
  # Get all the operations from the frozen graph. 
  constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
  for constant_op in constant_ops:
    #print(constant_op.name)
    if constant_op.name == 'final_retrain_ops/weights/final_weights':
        print(constant_op)
        pre_last_layer_weights = sess.run(constant_op.outputs[0]) # This gives a 2048 dimensional numpy array.
        print(pre_last_layer_weights.shape)
