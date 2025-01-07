from django.db import models
from bson import ObjectId  # Ensure bson is available if MongoDB is used
from django.utils.timezone import now

# Function to generate a default ObjectId
# def generate_object_id():
#     return str(ObjectId())

# class Person(models.Model):
#     # Unique Object ID for MongoDB compatibility
#     object_id = models.CharField(max_length=24, default=generate_object_id, editable=False)

#     # Registration number
#     # reg_no = models.CharField(max_length=50, null=True, blank=True)

#     # Name field
#     name = models.CharField(max_length=100)

#     # Role field
#     role = models.CharField(max_length=100)

#     # Wearing ID card field
#     wearing_id_card = models.BooleanField(default=False)

#     # Location field
#     location = models.CharField(max_length=100)

#     # Image field
#     image = models.ImageField(upload_to='images/', null=True, blank=True)

#     # Created at timestamp (automatically set at creation)
#     created_at = models.DateTimeField(auto_now_add=True)

#     # Custom Time field (using current UTC time)
#     time = models.DateTimeField(default=now)

#     person_id = models.CharField(max_length=24)

#     class Meta:
#         db_table = 'DetectionDB'  # Specify collection/table name for MongoDB or relational DB

#     def __str__(self):
#         return f"{self.name} - {self.role}"
