# from flask import Flask, request, render_template, Response
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import math
# import time

# app = Flask(__name__)
# device = torch.device("cpu")
# MODEL_PATH = "weights/last.pth.tar"

# # ------------------ تعریف مدل SRCNN ------------------ #
# class SRCNN(nn.Module):
#     def __init__(self) -> None:
#         super(SRCNN, self).__init__()
        
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
#             nn.ReLU(True)
#         )

#         self.map = nn.Sequential(
#             nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
#             nn.ReLU(True)
#         )

#         self.extra = nn.Sequential(
#             nn.Conv2d(32, 16, (3, 3), (1, 1), (1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
#             nn.ReLU(True),
#             nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
#             nn.ReLU(True)
#         )

#         self.reconstruction = nn.Sequential(
#             nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),  
#             nn.ReLU(True),
#             nn.Conv2d(16, 1, (5, 5), (1, 1), (2, 2))   
#         )

#         self._initialize_weights()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.features(x)
#         out = self.map(out)
#         out = self.extra(out)
#         out = self.reconstruction(out)
#         return out

#     def _initialize_weights(self) -> None:
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d):
#                 nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
#                 nn.init.zeros_(module.bias.data)

# # ------------------ پیش‌پردازش تصویر ------------------ #
# def preprocess_image(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = gray.astype(np.float32) / 255.0
#     tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#     return tensor

# # ------------------ ارتقای تصویر ------------------ #
# def enhance_image(img):
#     input_tensor = preprocess_image(img).to(device)

#     model.eval()
#     with torch.no_grad():
#         start = time.time()
#         output = model(input_tensor)
#         print("Time Taken: %.4f sec" % (time.time() - start))

#     output_image = output.squeeze(0).squeeze(0).cpu().numpy()
#     output_image = np.clip(output_image, 0, 1) * 255.0
#     output_image = output_image.astype(np.uint8)
#     output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)  # برای سازگاری با JPEG
#     return output_image

# # ------------------ صفحه اصلی ------------------ #
# @app.route('/')
# def index():
#     return render_template('index.html')

# # ------------------ آپلود تصویر ------------------ #
# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files['image']
#     img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

#     enhanced_img = enhance_image(img)

#     _, encoded_img = cv2.imencode('.jpg', enhanced_img)
#     return Response(encoded_img.tobytes(), mimetype='image/jpeg')

# # ------------------ اجرای اپلیکیشن ------------------ #
# if __name__ == '__main__':
#     model = SRCNN().to(device)
#     checkpoint = torch.load(MODEL_PATH, map_location=device)
#     model.load_state_dict(checkpoint['state_dict'])
#     app.run(debug=True)































from flask import Flask, request, render_template, Response
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import time

app = Flask(__name__)
device = torch.device("cpu")
MODEL_PATH = "weights/last.pth.tar"

# ------------------ تعریف مدل SRCNN ------------------ #
class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True)
        )

        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True)
        )

        self.extra = nn.Sequential(
            nn.Conv2d(32, 16, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True)
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),  
            nn.ReLU(True),
            nn.Conv2d(16, 1, (5, 5), (1, 1), (2, 2))   
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.extra(out)
        out = self.reconstruction(out)
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

# ------------------ پیش‌پردازش تصویر ------------------ #
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import time
from imgproc import bgr2ycbcr, ycbcr2bgr  # اضافه کن این دو تابع رو

# ...

def preprocess_image(img):
    # ورودی BGR با مقدار 0-255
    img = img.astype(np.float32) / 255.0
    # تبدیل به YCbCr
    img_ycbcr = bgr2ycbcr(img, only_use_y_channel=False)
    y, cb, cr = cv2.split(img_ycbcr)

    # فقط کانال Y برای مدل میفرستیم
    y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    return y_tensor, cb, cr

def enhance_image(img):
    input_tensor, cb, cr = preprocess_image(img)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)
        print("Time Taken: %.4f sec" % (time.time() - start))

    output_image = output.squeeze(0).squeeze(0).cpu().numpy()
    output_image = np.clip(output_image, 0, 1)

    # ترکیب کانال Y بازسازی شده با کانال‌های اصلی
    sr_ycbcr = cv2.merge([output_image, cb, cr])
    sr_bgr = ycbcr2bgr(sr_ycbcr)
    sr_bgr = np.clip(sr_bgr * 255.0, 0, 255).astype(np.uint8)

    # برای محاسبه PSNR و SSIM از کانال Y استفاده کن
    original_y = input_tensor.squeeze().cpu().numpy()
    psnr = peak_signal_noise_ratio(original_y, output_image, data_range=1.0)
    ssim = structural_similarity(original_y, output_image, data_range=1.0)

    return sr_bgr, psnr, ssim



# ------------------ صفحه اصلی ------------------ #
@app.route('/')
def index():
    return render_template('index.html')

# ------------------ آپلود تصویر ------------------ #
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    enhanced_img, psnr, ssim = enhance_image(img)

    _, encoded_img = cv2.imencode('.jpg', enhanced_img)
    result = {
        'image': encoded_img.tobytes(),
        'psnr': psnr,
        'ssim': ssim
    }

    from flask import jsonify
    response = Response(result['image'], mimetype='image/jpeg')
    response.headers['PSNR'] = str(psnr)
    response.headers['SSIM'] = str(ssim)
    return response


# ------------------ اجرای اپلیکیشن ------------------ #
if __name__ == '__main__':
    model = SRCNN().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    app.run(debug=True)























