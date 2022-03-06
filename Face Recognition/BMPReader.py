import struct


class Pixel(object):
    __slots__ = ['r', 'g', 'b']

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


class Bitmap:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.pixels = []
        for i in range(h):
            self.pixels.append([])
            for j in range(w):
                self.pixels[i].append(Pixel(0, 0, 0))

    def SetPx(self, w, h, r, g, b):
        self.pixels[w][h] = Pixel(r, g, b)

    def GetPx(self, w, h):
        return self.pixels[w][h]


def LoadBmp(path):
    f = open(path, "rb")
    offset = 0

    # HEADER
    signature = struct.unpack('<h', f.read(2))[0]
    fileSize = struct.unpack('<i', f.read(4))[0]
    r1 = struct.unpack('<h', f.read(2))[0]
    r2 = struct.unpack('<h', f.read(2))[0]
    pixelOffset = struct.unpack('<i', f.read(4))[0]

    dibSize = struct.unpack('<i', f.read(4))[0]
    width = struct.unpack('<i', f.read(4))[0]
    height = struct.unpack('<i', f.read(4))[0]
    planes = struct.unpack('<h', f.read(2))[0]
    bytesPerPixel = struct.unpack('<h', f.read(2))[0]
    compression = struct.unpack('<i', f.read(4))[0]

    imageSize = struct.unpack('<i', f.read(4))[0]
    horzPPM = struct.unpack('<i', f.read(4))[0]
    vertPPM = struct.unpack('<i', f.read(4))[0]
    palette = struct.unpack('<i', f.read(4))[0]
    colors = struct.unpack('<i', f.read(4))[0]
    # print(width, height)
    # PIXELS
    bitmap = Bitmap(width, height)
    padding = 4 - (3 * width) % 4
    if padding == 4:
        padding = 0
    for i in reversed(range(height)):
        for j in range(width):
            b = struct.unpack('<B', f.read(1))[0]
            g = struct.unpack('<B', f.read(1))[0]
            r = struct.unpack('<B', f.read(1))[0]
            bitmap.SetPx(i, j, r, g, b)
        for p in range(padding):
            junk = struct.unpack('<B', f.read(1))[0]
    return bitmap


def SaveBmp(bmp, path):
    b = ""

    # HEADER
    b += struct.pack('<B', 66)  # 'b'
    b += struct.pack('<B', 77)  # 'm'
    padding = 4 - (3 * bmp.width) % 4
    if padding == 4:
        padding = 0
    b += struct.pack('<L', 3 * bmp.width * bmp.height + 54 + bmp.height * padding)  # file size
    b += struct.pack('<H', 0)  # r1
    b += struct.pack('<H', 0)  # r2
    b += struct.pack('<L', 54)  # pixel start
    b += struct.pack('<L', 40)  # dib size
    b += struct.pack('<L', bmp.width)  # width
    b += struct.pack('<L', bmp.height)  # height
    b += struct.pack('<H', 8)  # planes
    b += struct.pack('<H', 24)  # depth
    b += struct.pack('<L', 0)  # compression
    b += struct.pack('<L', 3 * bmp.width * bmp.height + bmp.height * padding)  # image size
    b += struct.pack('<L', 0)  # hors ppm
    b += struct.pack('<L', 0)  # vert ppm
    b += struct.pack('<L', 0)  # palette
    b += struct.pack('<L', 0)  # important colors

    # PIXELS
    for i in reversed(range(bmp.height)):
        for j in range(bmp.width):
            b += struct.pack('<B', bmp.pixels[i][j].b)
            b += struct.pack('<B', bmp.pixels[i][j].g)
            b += struct.pack('<B', bmp.pixels[i][j].r)
        for p in range(padding):
            b += chr(238)
    with open(path, 'wb') as f:
        f.write(b)