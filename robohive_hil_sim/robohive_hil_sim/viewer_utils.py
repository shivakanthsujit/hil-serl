

import cv2
import numpy as np
from robohive_multi.utils.data_utils import visualkey_to_info

class OpenCVViewer:
    def __init__(
            self,
            env,
    ):
        self.env = env
        self.rgb_keys = [k for k in self.env.visual_keys if "rgb" in k]
        self.key_infos = {k: visualkey_to_info(k) for k in self.rgb_keys}
    
    def __enter__(self):
        self.launch()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def launch(self):
        cv2img = np.concatenate([np.zeros((v.height, v.width, 3), dtype=np.uint8) for k, v in self.key_infos.items()], axis=-2)
        self.update_cv2_window(cv2img)
    
    def is_running(self):
        # Check if OpenCV window is open
        return cv2.getWindowProperty("visuals", cv2.WND_PROP_VISIBLE) is not None
    
    def sync(self):
        images = self.env.visual_dict.copy()
        cv2img = np.concatenate([images[k] for k in self.rgb_keys], axis=-2)
        self.update_cv2_window(cv2img)

    def close(self):
        cv2.destroyAllWindows()

    def update_cv2_window(self, cv2img):
        cv2.imshow("visuals", cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)