import urllib.request

url = 'https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth'
filename = 'models/openpose_model.pth'

urllib.request.urlretrieve(url, filename)