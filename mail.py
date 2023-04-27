
from email.message import EmailMessage
import ssl
import smtplib

def send_email(cont_dic, mail_receiver):

    mail_sender = 'isaacinfrastructure@gmail.com'
    mail_password = 'wakmwxfbaikayjcs'
    
    
    subject = 'Complaint Register'
    body = f"Potholes are identifed at location: {cont_dic['location']}. It's a {cont_dic['highway_type']} that contains {cont_dic['size']}. Take necessary actions"

    em = EmailMessage()
    em['From'] = mail_sender
    em['To'] = mail_receiver
    em['subject'] = subject

    em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(mail_sender, mail_password)
        smtp.sendmail(mail_sender, mail_receiver, em.as_string())




