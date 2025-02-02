from django.apps import AppConfig
import socket

class WebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'web'

    def get_ip_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def ready(self):
        ip = self.get_ip_address()
        # write the result in .env file
        with open('./.env', 'w') as f:
            f.write(f"IP='{ip}'")
