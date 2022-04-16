import sys

import cv2

sys.path.append(".")
from src.__main__ import CLI
from src.models import SteganographyModels, get_model
from src.utils.logging import Symbol

IMAGE_PATH = "tests/test_images/cat.jpg"
MESSAGE_PATH = "tests/test_images/casoar.jpeg"

cli = CLI()
image = cli._load_image(IMAGE_PATH)
message = cli._load_image(MESSAGE_PATH)

for model_name in [item.name for item in SteganographyModels]:
    model = get_model(model_name)
    try:
        encoded_image = model.encode_img(image=image, to_encode=message)
        cv2.imwrite("tests/benchmark_images/encode_img_" + model_name + ".jpg", encoded_image)
        decoded_image = model.decode_img(image=encoded_image)
        cv2.imwrite("tests/benchmark_images/decode_img_" + model_name + ".jpg", decoded_image)

    except NotImplementedError:
        print(f"{Symbol.SKIP} Skipped encode_img {model_name}")
        pass
    except AttributeError:
        print(f"{Symbol.SKIP} Skipped encode_img {model_name}")
        pass

message = "You shall not see"

for model_name in [item.name for item in SteganographyModels]:
    model = get_model(model_name)
    try:
        encoded_image = model.encode_str(image, message)
        cv2.imwrite("tests/benchmark_images/encode_str_" + model_name + ".jpg", encoded_image)
        decoded_str = model.decode_str(encoded_image)
        if decoded_str == message:
            print(f"{Symbol.SUCCESS} Message decoded successfully {model_name}")
        else:
            print(f"{Symbol.FAIL} Message not decoded successfully {model_name}")

    except NotImplementedError:
        print(f"{Symbol.SKIP} Skipped encode_str {model_name}")
        pass
    except AttributeError:
        print(f"{Symbol.SKIP} Skipped encode_str {model_name}")
        pass
