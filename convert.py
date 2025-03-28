from PIL import Image
import numpy as np

def convert_image_to_raw(image_path, output_path):
    # Открываем изображение
    with Image.open(image_path) as img:
        # Преобразуем изображение в RGB (если оно в другом формате)
        img = img.convert('L')

        # Получаем данные пикселей
        pixel_data = np.array(img)

        # Проверяем размерность изображения
        height, width = pixel_data.shape
        print(f"Image Size: {width}x{height}")

        # Сохраняем данные в формате RAW
        with open(output_path, 'wb') as raw_file:
            raw_file.write(pixel_data.tobytes())

    print(f"Image saved as RAW format: {output_path}")


def convert_image_to_raw_BW(image_path, output_path):
    # Открываем изображение
    with Image.open(image_path) as img:
        # Преобразуем изображение в RGB (если оно в другом формате)
        img = img.convert('L')

        # Получаем данные пикселей
        pixel_data = np.array(img)
        height, width = pixel_data.shape
        size = (height, width)

        
        threshold = 128
        bw_image = img.point(lambda x: 255 if x > threshold else 1)
        pixels = np.array(bw_image)
        print(f"Image Size: {width}x{height}")

        # Сохраняем данные в формате RAW
        with open(output_path, 'wb') as raw_file:
            raw_file.write(pixels.tobytes())

    print(f"Image saved as RAW format: {output_path}")
# Пример использования
input_image_path = 'Path'  # Замените на путь к вашему изображению
output_raw_path = 'Path'   # Замените на желаемое имя выходного файла

convert_image_to_raw_BW(input_image_path, output_raw_path)