const hasAccountCheckbox = document.getElementById('has_account');
const hasPasswordCheckbox = document.getElementById('has_password');
const accountField = document.getElementById('account_field');
const passwordField = document.getElementById('password_field');

const hasCaptchaCheckbox = document.getElementById('has_captcha');
const captchaField = document.getElementById('captcha_field');

hasAccountCheckbox.addEventListener('change', () => {
  accountField.classList.toggle('hidden', !hasAccountCheckbox.checked);
});

hasPasswordCheckbox.addEventListener('change', () => {
  passwordField.classList.toggle('hidden', !hasPasswordCheckbox.checked);
});

hasCaptchaCheckbox.addEventListener('change', () => {
  captchaField.classList.toggle('hidden', !hasCaptchaCheckbox.checked);
});

