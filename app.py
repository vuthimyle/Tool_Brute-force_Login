import os
import time
from flask import Flask, request, render_template, redirect, url_for, flash, Response, stream_with_context
from attacker import dictionary_atatck, bruteforce_attack

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/dictionary', methods=['GET', 'POST'])
def dictionary():
    if request.method == 'GET':
        return render_template('dict.html')
    else:
        ip_address = request.form.get('ip_address')
        headers_raw = request.form.get('headers')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
        }
        

        # Chuyển headers từ dạng chuỗi -> dict
        if headers_raw:
            for line in headers_raw.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()
                    
        has_account = 'has_account' in request.form
        has_password = 'has_password' in request.form
        
        
        account_file = request.files.get('account_file')
        password_file = request.files.get('password_file')
        
        account_file_content = account_file.read().decode('utf-8').split() if account_file else None
        password_file_content = password_file.read().decode('utf-8').split() if password_file else None
        
        account = [request.form.get('account')] if has_account else account_file_content
        password = [request.form.get('password')] if has_password else password_file_content
        
        has_captcha = 'has_captcha' in request.form
        captcha_path = request.form.get('captcha_path') if has_captcha else None
        
        filter = request.form.get('filter')
        
        # Attack
        results = dictionary_atatck(ip_address, headers, account, password, captcha_path, filter)
        print(f"Results: {results}")
        return render_template('result.html', results=results)
        
        
        
@app.route('/bruteforce', methods=['GET', 'POST'])
def bruteforce():
    if request.method == 'GET':
        return render_template('brute.html')
    else:
        ip_address = request.form.get('ip_address')
        headers_raw = request.form.get('headers')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
        }
        
        if headers_raw:
            for line in headers_raw.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()

        has_account = 'has_account' in request.form
        has_password = 'has_password' in request.form
        
        username_length = 0
        password_length = 0
        
        if has_account:
            account = request.form.get('account')
        else:
            account = None
            username_length = int(request.form.get('username_length'), 0)
            
        if has_password:
            password = request.form.get('password') 
        else:
            password = None
            password_length = int(request.form.get('password_length', 0))

        username_opts = [
            True if opt in request.form else False
            for opt in ['username_num', 'username_char', 'username_special']
        ]

        password_opts = [
            True if opt in request.form else False
            for opt in ['password_num', 'password_char', 'password_special']
        ]

        has_captcha = 'has_captcha' in request.form
        captcha_path = request.form.get('captcha_path') if has_captcha else None
        
        filter = request.form.get('filter')


        # print(f"IP Address: {ip_address}")
        # print(f"Headers: {headers}")
        # print(f"Account: {account}")
        # print(f"Password: {password}")
        # print(f"Username Options: {username_opts} username_length: {username_length}")
        # print(f"Password Options: {password_opts} password_length: {password_length}")
        # print(f"Captcha Path: {captcha_path}")
        
        # Hiển thị ra màn hình
        results = bruteforce_attack(
            ip_address, headers, account, password, 
            username_length, password_length, 
            username_opts, password_opts, captcha_path, filter
            )
        print(results)
        return render_template('result.html', results=results)
    

        

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
