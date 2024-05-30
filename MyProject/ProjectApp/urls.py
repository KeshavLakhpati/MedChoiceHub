from django.contrib import admin
from django.urls import path
from ProjectApp import views,hospitals_details, doctors_details, labs_details
from . import views

urlpatterns = [
    path("",views.index, name='ProjectApp'),
    path("about_us",views.about_us, name='about_us'),
    path("services",views.services, name='services'),
    path("emergency",views.emergency, name='emergency'),
    path("patient_registration",views.patient_registration, name='patient_registration'),
    path("doctor_registration",views.doc_registration, name='doctor_registration'),
    path("appointment",views.appointment, name='appointment'),
    path("laboratory",views.laboratory, name='laboratory'),
    path("lab-details", labs_details.lab_details, name='lab_details'),
    path("login",views.login, name='login'),
    path("hospitals",hospitals_details.hospital_details, name='hospitals'),
    path("doctors",doctors_details.doctors_details, name='doctors'),
    path("prediction", views.show_prediction, name='prediction'),
    path('book_appointment/', views.book_appointment, name='book_appointment'),
    path('fetch_doctor_list/', views.fetch_doctor_list, name='fetch_doctor_list'),
]