# Yolo_fire-STPM_warnning 基于yolov5的火灾邮件预警程序

安装anaconda创建Python 3.7.16的环境

在anconda prompt窗口使用cd命令进入项目文件夹

使用命令

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install IPython roboflow -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install pynput

Python SMTP发送邮件

SMTP（Simple Mail Transfer Protocol）即简单邮件传输协议,它是一组用于由源地址到目的地址传送邮件的规则，由它来控制信件的中转方式。
python的smtplib提供了一种很方便的途径发送电子邮件。它对smtp协议进行了简单的封装

首先安装使用pip 命令安装

pip install smtplib

第一步：开启QQ邮箱的SMTP服务

详情请看：https://blog.csdn.net/Coin_Collecter/article/details/129596488

成功获取授权码后
修改 SMTP_warning/AlEmail.py 中第21 22行处qq邮箱和自己的授权码即可运行成功

