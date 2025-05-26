from flask import Flask, render_template, request, redirect, url_for
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from test2 import analyze_video
from flask import request, redirect, flash
import smtplib
from email.mime.text import MIMEText
app = Flask(__name__, template_folder="templates", static_folder='static')

app.secret_key = '4b#z7g9*pl1!jdf@k9v&x3q2!rmc$0sa'
@app.route('/')
def home():
    return render_template('home.html', title="Home", custom_css='home')


UPLOAD_FOLDER = 'static/uploaded_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            return redirect(url_for('result', filename=filename))  # نمرر اسم الملف هنا
        else:
            return "No file uploaded", 400
    return render_template('upload.html', title="Upload Video", custom_css='upload')



@app.route("/result/")
def result():
    filename = request.args.get('filename')
    result = analyze_video(filename)
    if not filename:
        return "No video specified", 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    deepfake_rate, final_result = analyze_video(video_path)

    return render_template('result.html', filename=filename, deepfake_rate=deepfake_rate, final_result=final_result, custom_css='result')


@app.route('/how-it-work')
def howItWork():
    return render_template('howItWork.html', title="How It Work", custom_css='how-it-work')

@app.route('/aboutus')
def aboutUs():
    return render_template('about.html', title="About Us", custom_css='about')

@app.route('/contact')
def contact():
    return render_template('contact.html', title="contact", custom_css='contact')


@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        name = request.form['userName']
        user_email = request.form['userEmail']
        subject = request.form['subject']
        message = request.form['message']

        full_message = f"""\
You received a new message:

Name: {name}
Email: {user_email}
Subject: {subject}
Message:
{message}
"""

        sender_email = "malekco721@gmail.com"
        receiver_email = "mq7995154@gmail.com"
        app_password = "jlzb ulgx xtmo kdqs"

        msg = MIMEMultipart()
        msg['Subject'] = Header(f"Contact Form - {subject}", 'utf-8')
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Reply-To'] = user_email

        msg.attach(MIMEText(full_message, 'plain', 'utf-8'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())

        flash('Message sent successfully!', 'success')
        return redirect('/contact')

    except Exception as e:
        print(e)
        flash(f'Error sending message: {str(e)}', 'danger')
        return redirect('/contact')

if __name__ == '__main__':
    app.run(debug=True)