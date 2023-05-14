# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from c2_processing import c2_process
from test import test
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import shutil
import requests
import wget
import time
from astropy.io import fits

run_flag = 2  # 0为自动下载网页地址进行检测, 1为对本地fts文件进行检测, 2为对本地fts文件进行批检测

search_threshold = 0.2  # 彗星检测阈值, 阈值过低容易误检测
sleep_time = 60 * 10  # 10分钟检测一次nasa网站, 查看有无新照片上传 (seconds * minutes)
start_date = 220813  # 起始下载日期 (yymmdd)
fts_file_path = ''  # 本地检测文件路径
process_file_path = ''  # 本地预处理结果生成路径
output_path = ''  # 本地最终结果生成路径
batch_size = 5  # run_flag = 2 下的批检测数量, 必须>5

from_name = ''  # 邮件发送者名称
to_name = ''  # 邮件接收者名称
from_addr = ''  # 发送邮箱
password = ''  # POP3/SMTP服务授权码
to_addr = ''  # 接收邮箱
smtp_server = ''  # 邮箱服务器：默认qq邮箱
port_num = 465  # 邮箱服务器端口号：默认qq邮箱
subject = ''  # 邮件主题
# 邮件正文
html_msg = """
    <p>text</p>
    """


def create_interest_region(answer_txt_path, input_process_path, output_path):
    image_size = 200
    font_pad_size = 25
    font_size = 16
    os.makedirs(output_path) if not os.path.isdir(output_path) else None

    answer_txt = open(answer_txt_path)
    answer_txt_lines = answer_txt.readlines()
    answer_txt.close()
    if len(answer_txt_lines) == 0:
        print('没有检测到彗星')
        return -1
    max_confidence = 0
    max_confidence_index = -1
    for index, line in enumerate(answer_txt_lines):
        line = line.split('\n')[0]
        if len(line) == 0:
            continue
        now_confidence = float(line.split(',')[-1])
        if now_confidence > max_confidence:
            max_confidence = now_confidence
            max_confidence_index = index
    if max_confidence < search_threshold:
        print('检测出最大置信度 = {:.3f} < 当前检测阈值 = {}, 已过滤'.format(max_confidence, search_threshold))
        return -1

    select_line = answer_txt_lines[max_confidence_index].split('\n')[0]
    answer = select_line.split(',')

    fts_x_y_dict = dict()
    fts_x_y_dict['fts'] = []
    fts_x_y_dict['x'] = []
    fts_x_y_dict['y'] = []
    i = 0
    while i < len(answer) - 1:
        if answer[i] == '.':
            i += 1
            continue
        if 'fts' in answer[i]:
            fts_x_y_dict['fts'].append(answer[i].split('\\')[-1])
            fts_x_y_dict['x'].append(int(answer[i+1]))
            fts_x_y_dict['y'].append(int(answer[i+2]))
            # print(fts_x_y_dict['fts'][-1], fts_x_y_dict['x'][-1], fts_x_y_dict['y'][-1])
            i += 3
        else:
            raise ValueError('1')

    process_png_list = os.listdir(input_process_path)  # './real-time data/220813_process/'
    process_png_list = [i for i in process_png_list if 'medfilt.png' in i]
    process_png_list.sort()
    assert len(fts_x_y_dict['fts']) == len(process_png_list)

    for i in tqdm(range(len(process_png_list))):
        process_image = Image.open(input_process_path + process_png_list[i])
        process_image_np = np.array(process_image)
        process_image_pad_100 = np.uint8(np.zeros((1024 + image_size, 1024 + image_size, 4)) + 255)
        process_image_pad_100[image_size//2: 1024 + image_size//2, image_size//2: 1024 + image_size//2, :] = process_image_np
        x, y = round(fts_x_y_dict['x'][i]), round(fts_x_y_dict['y'][i])
        if not (0 <= x <= 1024 and 0 <= y <= 1024):
            continue
        r = ((x - 512) ** 2 + (y - 512) ** 2) ** 0.5
        if r < 200:
            continue
        choose_x, choose_y = 1024 + image_size - (x + image_size // 2), y + image_size // 2
        image_x_y = process_image_pad_100[choose_y - image_size // 2: choose_y + image_size // 2,
                                          choose_x - image_size // 2: choose_x + image_size // 2, :]
        image_x_y_pad = np.uint8(np.zeros((font_pad_size + image_size, image_size, 4)) + 0)
        image_x_y_pad[font_pad_size: font_pad_size + image_size, :, :] = image_x_y
        image_x_y = image_x_y_pad
        image_x_y = Image.fromarray(image_x_y)
        image_font = ImageFont.truetype('msyh.ttc', font_size)
        image_draw = ImageDraw.Draw(image_x_y)
        # 右上角转换为左上角 x->1024-x
        image_information = '{}: {},{}'.format(process_png_list[i].split('_C2')[0], 1024 - x, y)
        image_draw.text((0, 0), image_information, font=image_font, fill="#000000")
        image_x_y.save('{}{}.png'.format(output_path, image_information.replace(': ', '-')))
    return max_confidence


def send_email(attach_file_path):

    msg = MIMEMultipart()
    msg['From'] = Header('{}'.format(from_name))
    msg['To'] = Header(to_name)

    msg['Subject'] = Header(subject, 'utf-8')
    msg.attach(MIMEText(html_msg, 'html', 'utf-8'))

    png_file_list = os.listdir(attach_file_path)
    if len(png_file_list) == 0:
        return

    for png_file in png_file_list:
        att = MIMEText(open(attach_file_path + png_file, 'rb').read(), 'base64', 'utf-8')
        att["Content-Type"] = 'application/octet-stream'
        att["Content-Disposition"] = 'attachment; filename="{}"'.format(png_file)
        msg.attach(att)

    try:
        smtpobj = smtplib.SMTP_SSL(smtp_server)
        smtpobj.connect(smtp_server, port_num)
        smtpobj.login(from_addr, password)
        smtpobj.sendmail(from_addr, to_addr, msg.as_string())
        print("email sent successfully")
    except smtplib.SMTPException:
        print("send email failed")
    finally:
        smtpobj.quit()


def search_comet_from_fts_file(fts_file_path, process_file_path, output_path):
    c2_process(fts_path=fts_file_path, dst_path=process_file_path, rundiff=False, annotate_words=False)  # 预处理
    test(folder_in=fts_file_path, output_file=process_file_path + 'output.txt')  # 通过fts文件计算彗星位置
    flag = create_interest_region(answer_txt_path=process_file_path + 'output.txt', input_process_path=process_file_path,
                                  output_path=output_path)  # 将预处理结果进行彗星位置截取
    if flag == -1:
        print('没有检测到彗星或检测置信度过低，不发送邮件')
    else:
        send_email(output_path)


def search_comet_from_fts_file_batch_mode(fts_file_path, output_path, batch_size):
    assert batch_size >= 5
    fts_file_list = os.listdir(fts_file_path)
    fts_file_list = [i for i in fts_file_list if '.fts' in i]
    fts_file_list.sort()
    comet_data = fits.getdata(fts_file_path + fts_file_list[0], header=True)[1]['DATE-OBS'].replace('/', '')
    for i in range(len(fts_file_list) // batch_size + 1):
        batch_fts_file_list = fts_file_list[i * batch_size: (i + 1) * batch_size]
        if len(batch_fts_file_list) < 5:
            continue
        batch_output_path = '{}/{:03d}_{}_/'.format(output_path, i + 1, comet_data)
        os.makedirs(batch_output_path) if not os.path.isdir(batch_output_path) else None
        for file in batch_fts_file_list:
            if not os.path.exists(batch_output_path + file):
                shutil.copyfile(fts_file_path + file, batch_output_path + file)
        c2_process(fts_path=batch_output_path, dst_path=batch_output_path,
                   rundiff=False, annotate_words=False, batch_mode=True)  # 预处理
        test(folder_in=batch_output_path, output_file=batch_output_path + 'output.txt')  # 通过fts文件计算彗星位置
        max_confidence = create_interest_region(answer_txt_path=batch_output_path + 'output.txt',
                                                input_process_path=batch_output_path,
                                                output_path=batch_output_path)  # 将预处理结果进行彗星位置截取
        if max_confidence == -1:
            shutil.rmtree(batch_output_path)
        else:
            now_files = os.listdir(batch_output_path)
            for file in now_files:
                if 'medfilt.png' in file or 'fts' in file or 'txt' in file:
                    os.remove(batch_output_path + file)
            os.rename(batch_output_path, '{}{:.2f}/'.format(batch_output_path[:-1], max_confidence))
            send_email('{}{:.2f}/'.format(batch_output_path[:-1], max_confidence))


def download(url, filename):
    try:
        wget.download(url, out=filename)
    except:
        print('网络不佳, 重新下载...')
        download(url, filename)


def get_url_text(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # 如果状态码不是200，产生异常
        r.encoding = 'utf-8'  # 字符编码格式改成 utf-8
        text = r.text
    except:
        print('网页读取异常, 重新尝试')
        text = get_url_text(url)
    return text


def download_fts_file_from_nasa(root_path, url):

    fts_file_list = []
    url_text = get_url_text(url)
    url_text_lines = url_text.split('\n')
    for line in url_text_lines:
        if 'fts' in line:
            fts_file_list.append(line.split('href="')[-1].split('">')[0])
    date = url.split('/')[-3]
    dst_path = root_path + date + '/'  # './real-time data/220813/'
    os.makedirs(dst_path) if not os.path.isdir(dst_path) else None
    new_download_flag = False
    for fts_file in tqdm(fts_file_list, desc='download {} fts_file'.format(date)):
        if os.path.exists(dst_path + fts_file):
            continue
        download(url + fts_file, dst_path + fts_file)
        new_download_flag = True
    tmp_file_list = os.listdir(dst_path)
    tmp_file_list = [i for i in tmp_file_list if 'tmp' in i]
    if len(tmp_file_list) != 0:
        for tmp_file in tmp_file_list:
            os.remove(dst_path + tmp_file)
    return dst_path, len(fts_file_list), new_download_flag


def auto_download_from_nasa(root_url, root_download_path, start_date):
    total_date = []
    url_text = get_url_text(root_url)
    url_text_lines = url_text.split('\n')
    for line in url_text_lines:
        if 'folder.gif' in line:
            total_date.append(int(line.split('href="')[-1].split('/')[0]))
    last_date_fts_file_num = None
    for date in total_date:
        if date < start_date:
            continue
        date_url = root_url + str(date) + '/c2/'
        dst_path, last_date_fts_file_num, new_download_flag = download_fts_file_from_nasa(root_download_path, date_url)
        if last_date_fts_file_num >= 6 and new_download_flag:
            search_comet_from_fts_file(dst_path, dst_path[:-1] + '_process/', dst_path[:-1] + '_process_results/')
    return total_date[-1], last_date_fts_file_num


def auto_search_comet(root_url, root_download_path, start_date):
    while 1:
        last_date, last_date_file_num = auto_download_from_nasa(root_url, root_download_path, start_date)
        start_date = last_date
        time.sleep(sleep_time)


if __name__ == '__main__':
    if run_flag == 0:
        auto_search_comet(root_url='https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/',
                          root_download_path='./real-time data/',
                          start_date=start_date)
    elif run_flag == 1:
        search_comet_from_fts_file(fts_file_path=fts_file_path,
                                   process_file_path=process_file_path,
                                   output_path=output_path)
    elif run_flag == 2:
        search_comet_from_fts_file_batch_mode(fts_file_path=fts_file_path,
                                              output_path=output_path,
                                              batch_size=batch_size)
    else:
        raise ValueError('error run_flag')
