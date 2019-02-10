import os
basedir = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = os.urandom(24)

RECAPTCHA_PUBLIC_KEY = '6LfC6x4UAAAAAM_37yVtTCbWTvj2A-1RYO02l1Ay'
RECAPTCHA_PRIVATE_KEY = '6LfC6x4UAAAAACIDnsKT32rAJkIuU7DD6B7EFPN9'