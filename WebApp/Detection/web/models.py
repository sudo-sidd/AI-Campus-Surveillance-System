# web/models.py
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=100)
    wearing_id_card = models.BooleanField(default=False)
    location = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    time = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'web_person'  # Specify the MongoDB collection name
