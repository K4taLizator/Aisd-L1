import os
from struct import pack, unpack
import time
import numpy as np
import sys
import matplotlib.pyplot as plt

def progress_bar(iteration, total, length=40):
    # Вычисляем процент завершения
    percent = (iteration / total)
    # Вычисляем количество символов для заполнения прогресс-бара
    filled_length = int(length * percent)
    # Создаем строку прогресс-бара
    bar = '█' * filled_length + '-' * (length - filled_length)
    # Выводим прогресс-бар
    sys.stdout.write(f'\r|{bar}| {percent:.2%} Complete')
    sys.stdout.flush()


def lzw_compress(uncompressed):
    # Создаем словарь для хранения последовательностей
    dictionary = {chr(i): i for i in range(256)}  # Инициализируем словарь с символами ASCII
    dict_size = 256
    result = []

    w = ""
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            try:
                result.append(dictionary[w])
                # Добавляем новую последовательность в словарь
                dictionary[wc] = dict_size
                dict_size += 1
                w = c
            except:
                print(w)
                break

    # Обрабатываем оставшуюся последовательность
    if w:
        result.append(dictionary[w])

    return result
def lzw_decompress(compressed):
    # Восстанавливаем словарь
    dictionary = {i: chr(i) for i in range(256)}
    dict_size = 256
    w = chr(compressed[0])
    result = [w]

    for k in compressed[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError("Invalid compressed k: {}".format(k))

        result.append(entry)

        # Добавляем новую последовательность в словарь
        dictionary[dict_size] = w + entry[0]
        dict_size += 1

        w = entry

    return ''.join(result)

def RLE(inp):
    count = 0
    let = inp[0]
    out=""
    for i in inp:
        if let == i and count < 255:
            count+=1
        else:
            out+=f"{chr(count)}{let}"
            let = i
            count = 1
    if count != 255:
        out+=f"{chr(count)}{let}" 
    return out
def unRLE(inp):
    out=""
    for i in range(1, len(inp), 2):
        out += inp[i]*(ord(inp[i-1]))
    return out

def MTF(input_string):
    alphabet = [chr(i) for i in range(256)]  # Начинаем с пустого алфавита
    output_indices = []

    for char in input_string:
        if char in alphabet:
            # Если символ уже в алфавите, перемещаем его в начало
            output_indices.append(chr(alphabet.index(char)))
            alphabet.remove(char)
            alphabet.insert(0, char)
        else:
            output_indices.append(char)

    return output_indices  # Возвращаем индексы и текущий алфавит
def unMTF(indices):
    alphabet = [chr(i) for i in range(256)]
    output_string = []
    for index in indices:
        if index in alphabet:
            char = alphabet[ord(index)]
            output_string.append(char)
        # Перемещаем символ в начало алфавита
            alphabet.remove(char)
            alphabet.insert(0, char)
        else:
            output_string.append(index)

    return ''.join(output_string)

def build_suffix_array(s):
    n = len(s)
    suffixes = sorted((s[i:], i) for i in range(n))  # Сортируем суффиксы
    suffix_array = [suffix[1] for suffix in suffixes]  # Извлекаем индексы
    return suffix_array   
def build_L(s, suffix_array):
    n = len(s)
    L = [''] * n
    for i in range(n):
        # Находим индекс суффикса и берем последний символ
        suffix_index = suffix_array[i]
        L[i] = s[suffix_index - 1] if suffix_index > 0 else s[-1]  # Обрабатываем циклический случай
    return L
def BWT(s):
    s+=chr(255)
    suffix_array = build_suffix_array(s)  # Этап 1
    L = build_L(s, suffix_array)  # Этап 2
    return ''.join(L)  # Объединяем массив L в строку
def unBWT(inp):
    out = ""
    right = []
    base = set(inp)
    base = dict((i, 0) for i in base)
    for i in inp:
        right.append(f"{i}{base[i]}")
        base[i]+=1
    left = sorted(right, key = lambda x: x[0])
    np = right.index(f"{chr(255)}0")
    for _ in range(len(inp)):
        out+=left[np][0]
        np = right.index(left[np])
    return out[:-1]

def LZW(inp):
    base = dict((chr(i), i) for i in range(256))
    cur = "" 
    encode = []
    for i in inp:
        if len(encode)%10000 == 0:
            progress_bar(len(encode), len(inp))
        temp = cur + i
        if base.get(temp) != None:
            cur = temp
            continue
        else:
            encode.append(base[cur])
            base[temp] = len(base)
            cur = i
    if temp in base:
        encode.append(base[temp])
    out = []
    for data in encode:
        out.append(chr(data))
    return "".join(out)
def unLZW(encode):
    out = ""
    compressed_data = [unpack('>H', i)[0] for i in encode]
    base = dict([(i, chr(i)) for i in range(256)])
    temp = ""
    
    for code in compressed_data:
        if not (code in base):
            base[code] = temp + (temp[0])
        out += base[code]
        if not(len(temp) == 0):
            base[len(base)-1] = temp + (base[code][0])
        temp = base[code]
    return out

"""def LZSS(data):
    compressed_data = []
    i = 0
    while i < len(data):
        match = find_longest_match(data, i)
        if match:
            offset, length = match
            if i+length < len(data):
                compressed_data.append(f"{chr(offset)}{chr(length)}{data[i + length]}")
            else:
                compressed_data.append(f"{chr(offset)}{chr(length)}{data[len(data)-1]}")
            i += length+1
        else:
            compressed_data.append(f"{chr(0)}{chr(0)}{data[i]}")
            i += 1
    return "".join(compressed_data)
def find_longest_match(data, current_index):
    buf = 128
    window = 256
    end_index = min(current_index + buf, len(data))
    longest_match = (0, 0)  # offset, length
    for i in range(max(0, current_index - window), current_index):
        length = 0
        while (i + length < current_index) and (current_index + length < end_index) and (data[i + length] == data[current_index + length]):
            length += 1
        if length > longest_match[1]:
            longest_match = (current_index - i, length)
    return longest_match if longest_match[1] > 0 else None
def unLZSS(compressed_data):
    decompressed_data = []
    for i in range(2, len(compressed_data)):
        offset, length, next_char = ord(compressed_data[i-2]), ord(compressed_data[i-1]), compressed_data[i]
        if length > 0:
            start_index = len(decompressed_data) - offset
            for i in range(length):
                decompressed_data.append(decompressed_data[start_index + i-1])
        decompressed_data.append(next_char)
    return ''.join(decompressed_data)"""

def LZSS(data, buf):
    window_size = buf
    lookahead_buffer_size = buf
    compressed = []
    i = 0
    n = len(data)

    while i < n:
        match_offset = 0
        match_length = 0

        # Поиск самого длинного совпадения
        for j in range(max(0, i - window_size), i):
            length = 0
            while (length < lookahead_buffer_size and
                   i + length < n and
                   data[j + length] == data[i + length]):
                length += 1

            if length > match_length:
                match_length = length
                match_offset = i - j

        if match_length >= 3:  # минимальная длина совпадения
            offset = bin(match_offset)[2:]
            length = bin(match_length)[2:]
            char = "10"+"0"*(12-len(offset))+offset+"0"*(12-len(length))+length
            compressed.append(char)
            i += match_length
        else:
            if ord(data[i]) < 256:
                compressed.append("00"+"0"*(8-len(bin(ord(data[i]))[2:]))+bin(ord(data[i]))[2:])
            else:
                compressed.append("01"+"0"*(16-len(bin(ord(data[i]))[2:]))+bin(ord(data[i]))[2:])
            i += 1
        if len(compressed[-1])%2==1:
            print("GG ", i)
    out = "".join(compressed)
    out += "0" * (8-len(out)%8) if out[-1] == "1" else "1" * (8-len(out)%8)
    return "".join([chr(int(out[i: i+8], 2)) for i in range(0, len(out), 8)])

def unLZSS(compressed):
    decompressed = []
    binary=""
    for i in compressed:
        t1 = bin(ord(i))[2:]
        binary+="0"*(8-len(t1))+t1
    ending = binary[-1]
    for i in range(len(binary)-1, len(binary)-10, -1):
        if binary[i] == ending:
            continue
        else:
            binary = binary[:i+1]
            break


    i = 0

    while i < len(binary):
        if binary[i]+binary[i+1] == "00":
            decompressed.append(chr(int(binary[i+2:i+10], 2)))

            i+=10
        elif binary[i]+binary[i+1] == "01": 
            decompressed.append(chr(int(binary[i+2:i+18], 2)))

            i+=18
        elif binary[i]+binary[i+1] == "10": 
            temp = binary[i+2:i+26]
            offset = int(temp[:len(temp)-12], 2)
            length = int(temp[len(temp)-12:], 2)
            start = len(decompressed) - offset

            for j in range(length):
                decompressed.append(decompressed[start + j])
            i+=26
    return ''.join(decompressed)

import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    frequency = dict()
    for char in text:
        frequency[char] = frequency.get(char, 0) + 1

    # Создаем кучу для узлов
    priority_queue = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

def generate_codes(node, current_code="", codes=None):
    if codes is None:
        codes = {}
    if node is not None:
        if node.char is not None:
            codes[node.char] = current_code
        generate_codes(node.left, current_code + "0", codes)
        generate_codes(node.right, current_code + "1", codes)
    return codes

def Huffman(text):
    if not text:
        return "", None

    root = build_huffman_tree(text)
    huffman_codes = generate_codes(root)

    encoded_text = ''.join(huffman_codes[char] for char in text)
    return encoded_text, huffman_codes

def unHuffman(encoded_text, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    current_code = ""
    decoded_text = ""

    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""

    return decoded_text


def Entropy(inp):
    mas = ""
    
    out = []
    for j in [30000, 50000]:
        frequency = dict()
        for i in range(len(inp)//j):
            temd_d = BWT(inp[i*j: (i+1)*j if (i+1)*j < len(inp) else len(inp)-1])
            mas = MTF(temd_d)
            progress_bar(i, len(inp)//j)
            for char in mas:
                if char in frequency:
                    frequency[char] += 1
                else:
                    frequency[char] = 1
        temp = 0
        for k, v in frequency.items():
            temp += v/len(inp) * np.log2(v/len(inp))
        out.append(-temp)
        print(-temp)
    return out

# Пример использования
file = open("Path", 'r', encoding='ANSI').readlines()
#size = os.path.getsize("Path") #for LZSS research
s = ''.join(list(file))
data = s


count = 0
N = 2048
mas = []

with open('output.txt', 'w', encoding="UTF-8") as file:
    code_t = 0
    decode_t = 0
    st = time.perf_counter()
    end = time.perf_counter()
    for i in range(0, len(data)//N):
        if i%20==0:
            progress_bar(i, len(data)//N)
        st = time.perf_counter()
        temp = data[i*N: (i+1)*N]
        t1 = BWT(temp)
        t2= MTF(t1)
        #t3 = RLE(t2)
        t4, code = Huffman(t2)
        end = time.perf_counter()
        code_t += end-st

        st = time.perf_counter()
        q1 = unHuffman(t4, code)
        #q2 = unRLE(q1)
        q3 = unMTF(q1)
        q4 = unBWT(q3)
        #q4 = unHuffman(t1, code)
        end = time.perf_counter()
        decode_t += end-st
        #t_out = "".join(t1)
        """t_out = "".join(chr(int(t4[i: i+8], 2)) if i+8 < N else chr(int(t1[i: ], 2))   for i in range(0, N, 8))
        t_out += "".join(code)"""
        file.write(t4)
        file.write("".join(code))
        
        if(temp != q4):   
            print("GG error")
            print(temp)
            print(q4)
            break
    mas.append(os.path.getsize("out_path"))
    print(f"\nCode time: {code_t}")
    print(f"Decode time: {decode_t}")

"""
plt.figure()
plt.plot([300, 600, 1000, 1500, 2047], [0.9665867514407572, 1.0580753804729617, 1.1108904349778812, 1.143606074591399, 1.1503086311839448])
plt.plot([300, 600, 1000, 1500, 2047], [0.9665867514407572, 1.0580753804729617, 1.1108904349778812, 1.143606074591399, 1.1503086311839448], "*r")
plt.grid()
plt.title("Зависимость коэффициента сжатия от размера буфера LZSS")
plt.xlabel("Размер буфера")
plt.ylabel("Коэффициент сжатия")
plt.show()

"""
