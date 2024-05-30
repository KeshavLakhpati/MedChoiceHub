from django.db import models

class PatientRegister(models.Model):
    firstname = models.CharField(max_length=20)
    lastname = models.CharField(max_length=20)
    email = models.CharField(max_length=50)
    mobileNo = models.CharField(max_length=15)
    location = models.CharField(max_length=20)
    registerUsername = models.CharField(max_length=50)
    registerPassword = models.CharField(max_length=20)


class DoctorRegister(models.Model):
    firstname = models.CharField(max_length=20)
    lastname = models.CharField(max_length=20)
    qualification = models.CharField(max_length=20)
    specialization = models.CharField(max_length=20)
    mobileNo = models.CharField(max_length=15)
    location = models.CharField(max_length=20)
    email = models.CharField(max_length=50)
    registerUsername = models.CharField(max_length=50)
    registerPassword = models.CharField(max_length=20)


class Appointment(models.Model):
    fullname = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    locality = models.CharField(max_length=100)
    disease = models.CharField(max_length=100)
    specialization = models.CharField(max_length=100)
    


class Emergency(models.Model):
    fullname = models.CharField(max_length=100)
    email = models.EmailField()
    hospital= models.TextField()
    mobileNo = models.CharField(max_length=15)
    disease = models.CharField(max_length=100)

# Create your models here.
