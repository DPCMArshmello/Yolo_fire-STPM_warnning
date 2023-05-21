#coding=utf-8

import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading

import requests
import json
import codecs


class smtpEamil:
    """
    邮件发送系统工具类
    """
    def __init__(self):
        self.qqAccout = '2068962182@qq.com'  # 发送邮件邮箱账号
        self.qqCode = 'wbuqricxhfezbdge'  # 授权码
        self.smtp_server = 'smtp.qq.com'  #qq邮箱smtp服务器
        self.smtp_port = 465   #端口

        self.formName='Python邮件预警系统'
        self.toName='管理员'
        self.titleName='Python SMTP 邮件测试'

        self.stmp=smtplib.SMTP_SSL(self.smtp_server,self.smtp_port)
        self.stmp.login(self.qqAccout,self.qqCode)


    def asContent(self,content):
        """
        返回邮件发送的内容
        :param content:
        :return:
        """
        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = Header(self.formName, 'utf-8')
        message['To'] = Header(self.toName, 'utf-8')
        subject = self.titleName     #邮件标题
        message['Subject'] = Header(subject, 'utf-8')

        return message.as_string()

    def asImageContent(self,content,imageName):
        """
        发送图片文件
        :param content: 正文内容
        :param imageName:  图片名称,注意要在本目录下的图片
        :return:
        """
        msgRoot = MIMEMultipart('related')
        msgRoot['From'] = Header(self.formName, 'utf-8')
        msgRoot['To'] = Header(self.toName, 'utf-8')
        subject =  self.titleName
        msgRoot['Subject'] = Header(subject, 'utf-8')
        msgAlternative = MIMEMultipart('alternative')
        msgRoot.attach(msgAlternative)
        mail_msg = """ <p>Python 邮件发送测试...</p> <p><a href="http://blog.momoxiaoming.com">momoxiaoming博客</a></p> <p>图片演示：</p> <p><img src="cid:image1"></p> """
        msgAlternative.attach(MIMEText(mail_msg, 'html', 'utf-8'))  # 指定图片为当前目录
        fp = open('test.jpg', 'rb')  # 找到程序当前目录图片
        msgImage = MIMEImage(fp.read())
        fp.close()  # 定义图片 ID，在 HTML 文本中引用
        msgImage.add_header('Content-ID', '<image1>')
        msgRoot.attach(msgImage)

        return msgRoot.as_string()

    def asFilesContent(self,content,files):
        """
        发送附件内容
        :param content:  正文内容
        :param files:  要发送的附件名数组,附件必须在本程序目录
        :return:
        """
        message = MIMEMultipart()
        message['From'] = Header(self.formName, 'utf-8')
        message['To'] = Header(self.toName, 'utf-8')
        subject =  self.titleName
        message['Subject'] = Header(subject, 'utf-8')  # 邮件正文内容
        message.attach(MIMEText(content, 'plain', 'utf-8'))  # 构造附件1，传送当前目录下的 test.txt 文件


        for f in files:
            att = MIMEText(open(f, 'rb').read(), 'base64', 'utf-8')
            att["Content-Type"] = 'application/octet-stream'
            att["Content-Disposition"] = 'attachment; filename='+f  # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
            message.attach(att)

        return message.as_string()

    def sendTEamil(self,msg,reciverAdress=None):
        """
        发送纯文本邮件
        :param reciverAdress: 收件人邮箱
        :param msg:  发送的文本内容
        :return:
        """
        if reciverAdress==None:
            reciverAdress=self.qqAccout   #假如收件邮箱为空,默认发到发送邮箱账号




        try:
            self.stmp.sendmail(self.qqAccout, reciverAdress, msg)
        except Exception as e:
            print('邮件发送失败--'+str(e))
        print('邮件发送成功')

    def sendEmail(self,msg,reciverAdress=None):
        """
        开辟线程发送,防止阻塞主线程
        :param msg:
        :param reciverAdress:
        :return:
        """
        t=threading.Thread(target=self.sendTEamil,args=(msg,reciverAdress))
        t.start()

def send_mian():
    email = smtpEamil()

    email.sendEmail(email.asContent('可能发生火灾预警'))

if __name__=='__main__':
    email = smtpEamil()

    #发送纯文本图片
    email.sendEmail(email.asContent('可能发生火灾预警'))
    #发送带图片邮件
    # email.sendEmail(email.asImageContent('以下是测试图片','test.jpg'))
    # #发送带附件邮件
    # email.sendEmail(email.asFilesContent('以下附件:',['test1.txt','test2.txt']))















