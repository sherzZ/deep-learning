
import time
t1 = time.time()

import sys
import os
import _io
from collections import namedtuple
from PIL import Image

class Nude(object):

    Skin = namedtuple("Skin", "id skin region x y")

    def __init__(self, path_or_image):
        # �� path_or_image Ϊ Image.Image ���͵�ʵ����ֱ�Ӹ�ֵ
        if isinstance(path_or_image, Image.Image):
            self.image = path_or_image
        # �� path_or_image Ϊ str ���͵�ʵ������ͼƬ
        elif isinstance(path_or_image, str):
            self.image = Image.open(path_or_image)

        # ���ͼƬ������ɫͨ��
        bands = self.image.getbands()
        # �ж��Ƿ�Ϊ��ͨ��ͼƬ��Ҳ���Ҷ�ͼ�������򽫻Ҷ�ͼת��Ϊ RGB ͼ
        if len(bands) == 1:
            # �½���ͬ��С�� RGB ͼ��
            new_img = Image.new("RGB", self.image.size)
            # �����Ҷ�ͼ self.image �� RGBͼ new_img.paste ��PIL �Զ�������ɫͨ��ת����
            new_img.paste(self.image)
            f = self.image.filename
            # �滻 self.image
            self.image = new_img
            self.image.filename = f

        # �洢��Ӧͼ���������ص�ȫ�� Skin ����
        self.skin_map = []
        # ��⵽��Ƥ������Ԫ�ص�������ΪƤ������ţ�Ԫ�ض��ǰ���һЩ Skin ������б�
        self.detected_regions = []
        # Ԫ�ض��ǰ���һЩ int ��������ţ����б�
        # ��ЩԪ���е�����Ŵ���������Ǵ��ϲ�������
        self.merge_regions = []
        # ���Ϻ��Ƥ������Ԫ�ص�������ΪƤ������ţ�Ԫ�ض��ǰ���һЩ Skin ������б�
        self.skin_regions = []
        # ����ϲ�������Ƥ�����������ţ���ʼ��Ϊ -1
        self.last_from, self.last_to = -1, -1
        # ɫ��ͼ���жϽ��
        self.result = None
        # ����õ�����Ϣ
        self.message = None
        # ͼ����
        self.width, self.height = self.image.size
        # ͼ��������
        self.total_pixels = self.width * self.height

    def resize(self, maxwidth=1000, maxheight=1000):
        """
        ��������߰���������ͼƬ��С��
        ע�⣺�����Ӱ�����㷨�Ľ��

        ���û�б仯���� 0
        ԭ��ȴ��� maxwidth ���� 1
        ԭ�߶ȴ��� maxheight ���� 2
        ԭ��ߴ��� maxwidth, maxheight ���� 3

        maxwidth - ͼƬ�����
        maxheight - ͼƬ���߶�
        ���ݲ���ʱ����������Ϊ False ������
        """
        # �洢����ֵ
        ret = 0
        if maxwidth:
            if self.width > maxwidth:
                wpercent = (maxwidth / self.width)
                hsize = int((self.height * wpercent))
                fname = self.image.filename
                # Image.LANCZOS ���ز����˲��������ڿ����
                self.image = self.image.resize((maxwidth, hsize), Image.LANCZOS)
                self.image.filename = fname
                self.width, self.height = self.image.size
                self.total_pixels = self.width * self.height
                ret += 1
        if maxheight:
            if self.height > maxheight:
                hpercent = (maxheight / float(self.height))
                wsize = int((float(self.width) * float(hpercent)))
                fname = self.image.filename
                self.image = self.image.resize((wsize, maxheight), Image.LANCZOS)
                self.image.filename = fname
                self.width, self.height = self.image.size
                self.total_pixels = self.width * self.height
                ret += 2
        return ret

    # ��������
    def parse(self):
        # ������н�������ر�����
        if self.result is not None:
            return self
        # ���ͼƬ������������
        pixels = self.image.load()
        # ����ÿ������
        for y in range(self.height):
            for x in range(self.width):
                # �õ����ص� RGB ����ͨ����ֵ
                # [x, y] �� [(x,y)] �ļ��д��
                r = pixels[x, y][0]   # red
                g = pixels[x, y][1]   # green
                b = pixels[x, y][2]   # blue
                # �жϵ�ǰ�����Ƿ�Ϊ��ɫ����
                isSkin = True if self._classify_skin(r, g, b) else False
                # ��ÿ�����ط���Ψһ id ֵ��1, 2, 3...height*width��
                # ע�� x, y ��ֵ���㿪ʼ
                _id = x + y * self.width + 1
                # Ϊÿ�����ش���һ����Ӧ�� Skin ���󣬲���ӵ� self.skin_map ��
                self.skin_map.append(self.Skin(_id, isSkin, None, x, y))
                # ����ǰ���ز�Ϊ��ɫ���أ������˴�ѭ��
                if not isSkin:
                    continue

                # �����Ͻ�Ϊԭ�㣬��������Ϊ���� *����ǰ����Ϊ���� ^����ô�໥λ�ù�ϵͨ������ͼ
                # ***
                # *^

                # �������������������б����˳��Ϊ�ɴ�С��˳��ı���Ӱ��
                # ע�� _id �Ǵ� 1 ��ʼ�ģ���Ӧ���������� _id-1
                check_indexes = [_id - 2, # ��ǰ�����󷽵�����
                                 _id - self.width - 2,  # ��ǰ�������Ϸ�������
                                 _id - self.width - 1,  # ��ǰ���ص��Ϸ�������
                                 _id - self.width]  # ��ǰ�������Ϸ�������
                # ������¼���������з�ɫ�������ڵ�����ţ���ʼ��Ϊ -1
                region = -1
                # ����ÿһ���������ص�����
                for index in check_indexes:
                    # ���������������ص� Skin ����û��������ѭ��
                    try:
                        self.skin_map[index]
                    except IndexError:
                        break
                    # ����������Ϊ��ɫ���أ�
                    if self.skin_map[index].skin:
                        # �����������뵱ǰ���ص� region ��Ϊ��Чֵ���Ҷ��߲�ͬ������δ�����ͬ�ĺϲ�����
                        if (self.skin_map[index].region != None and
                                region != None and region != -1 and
                                self.skin_map[index].region != region and
                                self.last_from != region and
                                self.last_to != self.skin_map[index].region) :
                            # ��ô���������������ĺϲ�����
                            self._add_merge(region, self.skin_map[index].region)
                        # ��¼�������������ڵ������
                        region = self.skin_map[index].region
                # �����������������غ��� region �Ե��� -1��˵�������������ض����Ƿ�ɫ����
                if region == -1:
                    # ��������Ϊ�µ�����ţ�ע��Ԫ���ǲ��ɱ����ͣ�����ֱ�Ӹ�������
                    _skin = self.skin_map[_id - 1]._replace(region=len(self.detected_regions))
                    self.skin_map[_id - 1] = _skin
                    # ���˷�ɫ�����������򴴽�Ϊ������
                    self.detected_regions.append([self.skin_map[_id - 1]])
                # region ������ -1 ��ͬʱ������ None��˵���������Ϊ��Чֵ�����ڷ�ɫ����
                elif region != None:
                    # �������ص�����Ÿ���Ϊ������������ͬ
                    _skin = self.skin_map[_id - 1]._replace(region=region)
                    self.skin_map[_id - 1] = _skin
                    # ���������������б�����Ӵ�����
                    self.detected_regions[region].append(self.skin_map[_id - 1])
        # �����������ϲ����񣬺ϲ�����������洢�� self.skin_regions
        self._merge(self.detected_regions, self.merge_regions)
        # ����Ƥ�����򣬵õ��ж����
        self._analyse_regions()
        return self


    # self.merge_regions ��Ԫ�ض��ǰ���һЩ int ��������ţ����б�
    # self.merge_regions ��Ԫ���е�����Ŵ���������Ǵ��ϲ�������
    # ����������ǽ��������ϲ����������ӵ� self.merge_regions ��
    def _add_merge(self, _from, _to):
        # ��������Ÿ�ֵ��������
        self.last_from = _from
        self.last_to = _to

        # ��¼ self.merge_regions ��ĳ������ֵ����ʼ��Ϊ -1
        from_index = -1
        # ��¼ self.merge_regions ��ĳ������ֵ����ʼ��Ϊ -1
        to_index = -1


        # ����ÿ�� self.merge_regions ��Ԫ��
        for index, region in enumerate(self.merge_regions):
            # ����Ԫ���е�ÿ�������
            for r_index in region:
                if r_index == _from:
                    from_index = index
                if r_index == _to:
                    to_index = index

        # ����������Ŷ������� self.merge_regions ��
        if from_index != -1 and to_index != -1:
            # �������������ŷֱ�����������б���
            # ��ô�ϲ��������б�
            if from_index != to_index:
                self.merge_regions[from_index].extend(self.merge_regions[to_index])
                del(self.merge_regions[to_index])
            return

        # ����������Ŷ��������� self.merge_regions ��
        if from_index == -1 and to_index == -1:
            # �����µ�������б�
            self.merge_regions.append([_from, _to])
            return
        # ���������������һ�������� self.merge_regions ��
        if from_index != -1 and to_index == -1:
            # ���������� self.merge_regions �е��Ǹ������
            # ��ӵ���һ����������ڵ��б�
            self.merge_regions[from_index].append(_to)
            return
        # ���������ϲ������������һ�������� self.merge_regions ��
        if from_index == -1 and to_index != -1:
            # ���������� self.merge_regions �е��Ǹ������
            # ��ӵ���һ����������ڵ��б�
            self.merge_regions[to_index].append(_from)
            return

    # �ϲ��úϲ���Ƥ������
    def _merge(self, detected_regions, merge_regions):
        # �½��б� new_detected_regions 
        # ��Ԫ�ؽ��ǰ���һЩ�������ص� Skin ������б�
        # new_detected_regions ��Ԫ�ؼ�����Ƥ������Ԫ������Ϊ�����
        new_detected_regions = []

        # �� merge_regions �е�Ԫ���е�����Ŵ������������ϲ�
        for index, region in enumerate(merge_regions):
            try:
                new_detected_regions[index]
            except IndexError:
                new_detected_regions.append([])
            for r_index in region:
                new_detected_regions[index].extend(detected_regions[r_index])
                detected_regions[r_index] = []

        # ���ʣ�µ�����Ƥ������ new_detected_regions
        for region in detected_regions:
            if len(region) > 0:
                new_detected_regions.append(region)

        # ���� new_detected_regions
        self._clear_regions(new_detected_regions)

    # Ƥ������������
    # ֻ��������������ָ��������Ƥ������
    def _clear_regions(self, detected_regions):
        for region in detected_regions:
            if len(region) > 30:
                self.skin_regions.append(region)

    # ��������
    def _analyse_regions(self):
        # ���Ƥ������С�� 3 ��������ɫ��
        if len(self.skin_regions) < 3:
            self.message = "Less than 3 skin regions ({_skin_regions_size})".format(
                _skin_regions_size=len(self.skin_regions))
            self.result = False
            return self.result

        # ΪƤ����������
        self.skin_regions = sorted(self.skin_regions, key=lambda s: len(s),
                                   reverse=True)

        # ����Ƥ����������
        total_skin = float(sum([len(skin_region) for skin_region in self.skin_regions]))

        # ���Ƥ������������ͼ��ı�ֵС�� 15%����ô����ɫ��ͼƬ
        if total_skin / self.total_pixels * 100 < 15:
            self.message = "Total skin percentage lower than 15 ({:.2f})".format(total_skin / self.total_pixels * 100)
            self.result = False
            return self.result

        # ������Ƥ������С����Ƥ������� 45%������ɫ��ͼƬ
        if len(self.skin_regions[0]) / total_skin * 100 < 45:
            self.message = "The biggest region contains less than 45 ({:.2f})".format(len(self.skin_regions[0]) / total_skin * 100)
            self.result = False
            return self.result

        # Ƥ�������������� 60��������ɫ��ͼƬ
        if len(self.skin_regions) > 60:
            self.message = "More than 60 skin regions ({})".format(len(self.skin_regions))
            self.result = False
            return self.result

        # �������Ϊɫ��ͼƬ
        self.message = "Nude!!"
        self.result = True
        return self.result

    # �������صķ�ɫ��⼼��
    def _classify_skin(self, r, g, b):
        # ����RGBֵ�ж�
        rgb_classifier = r > 95 and \
            g > 40 and g < 100 and \
            b > 20 and \
            max([r, g, b]) - min([r, g, b]) > 15 and \
            abs(r - g) > 15 and \
            r > g and \
            r > b
        # ���ݴ����� RGB ֵ�ж�
        nr, ng, nb = self._to_normalized(r, g, b)
        norm_rgb_classifier = nr / ng > 1.185 and \
            float(r * b) / ((r + g + b) ** 2) > 0.107 and \
            float(r * g) / ((r + g + b) ** 2) > 0.112

        # HSV ��ɫģʽ�µ��ж�
        h, s, v = self._to_hsv(r, g, b)
        hsv_classifier = h > 0 and \
            h < 35 and \
            s > 0.23 and \
            s < 0.68

        # YCbCr ��ɫģʽ�µ��ж�
        y, cb, cr = self._to_ycbcr(r, g,  b)
        ycbcr_classifier = 97.5 <= cb <= 142.5 and 134 <= cr <= 176

        # Ч�����Ǻܺã�����Ĺ�ʽ
        # return rgb_classifier or norm_rgb_classifier or hsv_classifier or ycbcr_classifier
        return ycbcr_classifier

    def _to_normalized(self, r, g, b):
        if r == 0:
            r = 0.0001
        if g == 0:
            g = 0.0001
        if b == 0:
            b = 0.0001
        _sum = float(r + g + b)
        return [r / _sum, g / _sum, b / _sum]

    def _to_ycbcr(self, r, g, b):
        # ��ʽ��Դ��
        # http://stackoverflow.com/questions/19459831/rgb-to-ycbcr-conversion-problems
        y = .299*r + .587*g + .114*b
        cb = 128 - 0.168736*r - 0.331364*g + 0.5*b
        cr = 128 + 0.5*r - 0.418688*g - 0.081312*b
        return y, cb, cr

    def _to_hsv(self, r, g, b):
        h = 0
        _sum = float(r + g + b)
        _max = float(max([r, g, b]))
        _min = float(min([r, g, b]))
        diff = float(_max - _min)
        if _sum == 0:
            _sum = 0.0001

        if _max == r:
            if diff == 0:
                h = sys.maxsize
            else:
                h = (g - b) / diff
        elif _max == g:
            h = 2 + ((g - r) / diff)
        else:
            h = 4 + ((r - g) / diff)

        h *= 60
        if h < 0:
            h += 360

        return [h, 1.0 - (3.0 * (_min / _sum)), (1.0 / 3.0) * _max]

    def inspect(self):
        _image = '{} {} {}��{}'.format(self.image.filename, self.image.format, self.width, self.height)
        return "{_image}: result={_result} message='{_message}'".format(_image=_image, _result=self.result, _message=self.message)

    # ����Դ�ļ�Ŀ¼����ͼƬ�ļ�����Ƥ��������ӻ�
    def showSkinRegions(self):
        # δ�ó����ʱ��������
        if self.result is None:
            return
        # Ƥ�����ص� ID �ļ���
        skinIdSet = set()
        # ��ԭͼ��һ�ݿ���
        simage = self.image
        # ��������
        simageData = simage.load()

        # ��Ƥ�����ص� id ���� skinIdSet
        for sr in self.skin_regions:
            for pixel in sr:
                skinIdSet.add(pixel.id)
        # ��ͼ���е�Ƥ��������Ϊ��ɫ��������Ϊ��ɫ
        for pixel in self.skin_map:
            if pixel.id not in skinIdSet:
                simageData[pixel.x, pixel.y] = 0, 0, 0
            else:
                simageData[pixel.x, pixel.y] = 255, 255, 255
        # Դ�ļ�����·��
        filePath = os.path.abspath(self.image.filename)
        # Դ�ļ�����Ŀ¼
        fileDirectory = os.path.dirname(filePath) + '/'
        # Դ�ļ��������ļ���
        fileFullName = os.path.basename(filePath)
        # ����Դ�ļ��������ļ����õ��ļ�������չ��
        fileName, fileExtName = os.path.splitext(fileFullName)
        # ����ͼƬ
        simage.save('{}{}_{}{}'.format(fileDirectory, fileName,'Nude' if self.result else 'Normal', fileExtName))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Detect nudity in images.')
    parser.add_argument('files', metavar='image', nargs='+',
                        help='Images you wish to test')
    parser.add_argument('-r', '--resize', action='store_true',
                        help='Reduce image size to increase speed of scanning')
    parser.add_argument('-v', '--visualization', action='store_true',
                        help='Generating areas of skin image')

    args = parser.parse_args()

    for fname in args.files:
        if os.path.isfile(fname):
            n = Nude(fname)
            if args.resize:
                n.resize(maxheight=800, maxwidth=600)
            n.parse()
            if args.visualization:
                n.showSkinRegions()
            print(n.result, n.inspect())
        else:
            print(fname, "is not a file")
    t2 = time.time()
    print t2-t1    