import sys

from darknet import Darknet
from bisenetv2 import BiSeNetV2
from mflops import get_model_compute_info

if __name__ == '__main__':
    ost = sys.stdout
    
    try:
        # darknet
        print("\nDarknet")
        model_def = "gesture5_v8.cfg"
        model = Darknet(model_def)  
    
        flops, mac, params = get_model_compute_info(model, (3, 224, 224))
    except:
        print("Can't load darknet.")
    
    try:
        # bisenet
        print("\nBiSeNetV2")
        model = BiSeNetV2(19, output_aux=False)
        flops, mac, params = get_model_compute_info(model, (3, 1024, 2048))
    except:
        print("Can't load bisenet.")