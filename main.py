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

from_name = 'Comet_Bot'  # 邮件发送者名称
to_name = 'Z. Chen'  # 邮件接收者名称
from_addr = 'time34@163.com'  # 发送邮箱
password = 'MBTJNFOIKZPVGILK'  # POP3/SMTP服务授权码
to_addr = '1583797321@qq.com'  # 接收邮箱
smtp_server = 'smtp.163.com'  # 邮箱服务器：默认qq邮箱
port_num = 465  # 邮箱服务器端口号：默认qq邮箱
subject = 'Candidate Discovered!'  # 邮件主题
# 邮件正文
html_msg = """
    <p>Candidate Found!</p>
    """

def create_interest_region(answer_txt_path, input_process_path, output_path):
    default_index = 1
    image_size = 200
    font_pad_size = 25
    font_size = 16
    os.makedirs(output_path) if not os.path.isdir(output_path) else None
    answer_txt = np.loadtxt(answer_txt_path, dtype='str')[default_index]
    answer = answer_txt.split(',')
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

    process_png_list = os.listdir(input_process_path)
    process_png_list = [i for i in process_png_list if 'png' in i]
    process_png_list.sort()
    assert len(fts_x_y_dict['fts']) == len(process_png_list)

    for i in tqdm(range(len(process_png_list))):
        process_image = Image.open(input_process_path + process_png_list[i])
        process_image_np = np.array(process_image)
        process_image_pad_100 = np.uint8(np.zeros((1024 + image_size, 1024 + image_size, 4)) + 255)
        process_image_pad_100[image_size//2: 1024 + image_size//2, image_size//2: 1024 + image_size//2, :] = process_image_np
        x, y = round(fts_x_y_dict['x'][i]), round(fts_x_y_dict['y'][i])
        choose_x, choose_y = 1024 + image_size - (x + image_size // 2), y + image_size // 2

        image_x_y = process_image_pad_100[choose_y - image_size // 2: choose_y + image_size // 2,
                                          choose_x - image_size // 2: choose_x + image_size // 2, :]
        image_x_y_pad = np.uint8(np.zeros((font_pad_size + image_size, image_size, 4)) + 0)
        image_x_y_pad[font_pad_size: font_pad_size + image_size, :, :] = image_x_y
        image_x_y = image_x_y_pad
        image_x_y = Image.fromarray(image_x_y)
        image_font = ImageFont.truetype('msyh.ttc', font_size)
        image_draw = ImageDraw.Draw(image_x_y)
        image_information = '{}: {},{}'.format(process_png_list[i].split('_C2')[0], x, y)
        image_draw.text((0, 0), image_information, font=image_font, fill="#000000")

        # image_x_y.show('1')
        # raise ValueError('1')
        image_x_y.save('{}{}.png'.format(output_path, image_information.replace(': ', '-')))
        '''
        box_width = 5
        line_width = 1
        for j in range(line_width):
            for _x in range(-box_width - j, box_width + j + 1):
                for _y in range(-box_width - j, box_width + j + 1):
                    if abs(_x) != box_width + j and abs(_y) != box_width + j:
                        continue
                    process_image_pad_100[choose_y + _y, choose_x + _x, :] = np.array([0, 0, 0, 0], dtype=np.uint8)
        image_show = Image.fromarray(process_image_pad_100)
        image_show.save(output_path + process_png_list[i])
        '''


def send_email(attach_file_path):

    msg = MIMEMultipart()
    msg['From'] = Header('{}'.format(from_name))
    msg['To'] = Header(to_name)

    msg['Subject'] = Header(subject, 'utf-8')
    msg.attach(MIMEText(html_msg, 'html', 'utf-8'))

    png_file_list = os.listdir(attach_file_path)
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
        print("Email sent successfully")
        
    except smtplib.SMTPException:
        print("Send email failed")

    finally:
        smtpobj.quit()


def main(fts_file_path, process_file_path, output_path):
    c2_process(fts_path=fts_file_path, dst_path=process_file_path, rundiff=False, annotate_words=False)
    test(folder_in=fts_file_path, output_file=process_file_path + 'output.txt')
    create_interest_region(answer_txt_path=process_file_path + 'output.txt', input_process_path=process_file_path,
                           output_path=output_path)
    send_email(output_path)


if __name__ == '__main__':
    main(fts_file_path='./data/cmt0030/',
         process_file_path='./data/cmt0030_process/',
         output_path='./data/cmt0030_process_results/')
