import cv2
import urllib.request

from segment_anything_onnx.inference import predict_masks


def load_image(uri):
    if( uri.startswith('https://') or uri.startswith('http://') ):
        req = urllib.request.urlopen(uri)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1) # 'Load it as it is'
        return img
    else: 
        return cv2.imread(uri)  # uri is just a local file path


encoder_model_path = './models/sam_vit_l_0b3195.encoder.onnx'
decoder_model_path = './models/sam_vit_l_0b3195.decoder.onnx'
image = load_image( './examples/laura.jpg' )
prompt = [
    { 'type': 'point', 'data': [1750, 300], 'label': 0 },
    { 'type': 'rectangle', 'data': [611, 655, 2712, 4500] }
]
options = {
    'show': True,
    'output': './output/laura-L.png'
}

predict_masks( encoder_model_path, decoder_model_path, image, prompt, options )

