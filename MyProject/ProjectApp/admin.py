from django.contrib import admin
from ProjectApp.models import PatientRegister, DoctorRegister, Appointment, Emergency

admin.site.register(PatientRegister)
admin.site.register(DoctorRegister)
admin.site.register(Appointment)
admin.site.register(Emergency)

# Register your models here.
