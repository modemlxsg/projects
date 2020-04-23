sess = rt.InferenceSession('./build/card_reg_v1.onnx')
# input_name = sess.get_inputs[0].name
# pred = sess.run(None, {input_name: img})[0]

# print(pred)