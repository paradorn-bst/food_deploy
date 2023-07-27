from method import preprocess_img, predict_img, report_img
import onnxruntime
import onnx

# base64 image test
from img_test import image

# load model
classifier = onnxruntime.InferenceSession("/model/efficientnet_classify_model.onnx")

# image base64 string
new_img = image

# transform image
new_img_transform = preprocess_img.preprocess_img(new_img)

# predict image
first_result, another_result = predict_img.predict_img(new_img_transform, classifier, recommend = 5)

# report the result
report = report_img.result_to_json(first_result, another_result)

# print result
print(report)