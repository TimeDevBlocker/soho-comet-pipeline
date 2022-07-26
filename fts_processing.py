from astropy.io import fits
import numpy as np
from PIL import Image
from c2_processing import c2_process
import os
import multiprocessing
import math


def test_fts_file(file_path):
    sample = fits.open(file_path)
    sample.info()
    image = sample[0].data  # (1024, 1024)
    print(np.max(image), np.min(image))
    image = np.uint8((image - np.min(image)) / (np.max(image) - np.min(image)) * 255)
    print(np.max(image), np.min(image))
    image = Image.fromarray(image)
    image.show('1')


def downloadgfs(filename):
    os.system(filename)


def download_fts_file(http, dst_file, index_range, max_multi=20):
    os.makedirs(dst_file) if not os.path.isdir(dst_file) else None
    download_cmd = []
    for index in range(index_range[0], index_range[1] + 1):
        download_cmd.append('wget -O {}{}.fts {}{}.fts'.format(dst_file, index, http, index))

    for i in range(math.ceil((index_range[1] - index_range[0] + 1) / (max_multi * 1.0))):
        p = multiprocessing.Pool(processes=max_multi)
        _ = [p.apply_async(func=downloadgfs, args=(cmd,)) for cmd in download_cmd[i * max_multi: (i + 1) * max_multi]]
        p.close()
        p.join()


if __name__ == '__main__':
    # test_fts_file('./challenge_data/cmt0001/22539961.fts')
    c2_process(fts_path='./challenge_data/cmt0030/', dst_path='./challenge_data/cmt0030_process/',
               rundiff=False, annotate_words=True)
    '''
    download_fts_file(http='https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/220716/c2/',
                      dst_file='D:/研究生/杂活/2022-7-23彗星搜索/data/',
                      index_range=(23884344, 23884468))  # (23884344, 23884468)
    '''



