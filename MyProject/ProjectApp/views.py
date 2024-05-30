
import csv
from django import forms
from ProjectApp.models import PatientRegister, DoctorRegister, Appointment, Emergency
from django.views.decorators.csrf import requires_csrf_token
from .disease_pred import predictDisease
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.urls import reverse
from .forms import AppointmentForm
from django.contrib import messages
@requires_csrf_token

def index(request):
    return render (request,"index.html")

def about_us(request):
    return render (request,"about_us.html")

def services(request):
    return render (request,"services.html")

def emergency(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        email = request.POST.get('email')
        hospital = request.POST.get('hosp_name')
        mobileNo = request.POST.get('mobileNo')
        disease = request.POST.get('disease')
        Users_table = Emergency(fullname=fullname, email=email, hospital = hospital, mobileNo=mobileNo, disease=disease)
        Users_table.save()
    return render (request,"emergency.html")

#def doctor_registration(request):
    #return render (request,"doctor_registration.html")

def patient_registration(request):
    if request.method == 'POST':
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        email = request.POST.get('email')
        mobileNo = request.POST.get('mobileNo')
        location = request.POST.get('location')
        registerUsername = request.POST.get('username')
        registerPassword = request.POST.get('password')
        Users_table = PatientRegister(firstname=firstname, lastname=lastname, email=email, mobileNo=mobileNo, location=location, registerUsername=registerUsername, registerPassword=registerPassword)
        Users_table.save()
    return render(request, 'patient_registration.html')

def doc_registration(request):
    if request.method == 'POST':
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        qualification = request.POST.get('qualification')
        specialization = request.POST.get('specialization')
        email = request.POST.get('email')
        mobileNo = request.POST.get('mobileNo')
        location = request.POST.get('location')
        registerUsername = request.POST.get('username')
        registerPassword = request.POST.get('password')
        Users_table = DoctorRegister(firstname=firstname, lastname=lastname, email=email, specialization = specialization, qualification = qualification, mobileNo=mobileNo, location=location, registerUsername=registerUsername, registerPassword=registerPassword)
        Users_table.save()
    return render(request, 'doctor_registration.html')

def login(request):
    return render (request,"login.html")


def appointment(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        disease = request.POST.get('disease')
        specialization = request.POST.get('specialization')
        locality = request.POST.get('locality')
        Users_table = Appointment(fullname=fullname, email=email, specialization = specialization, phone = phone, disease = disease, locality = locality)
        Users_table.save()
    return render (request,"appointment.html")

def laboratory(request):
    return render (request,"laboratory.html")

def hospitals(request):
    return render (request,"hospitals.html")

def show_prediction(request):
    if request.method == 'POST':
        symptoms = request.POST.get('selectedSymptoms', '')
        disease = predictDisease(symptoms)
        return render(request, 'prediction.html', {'disease': disease})
    return render(request, 'prediction.html')


# def fetch_doctor_details(request):
#     specialization = request.GET.get('specialization')
#     if specialization:
#         with open('E:/Medchoicehub/MyProject/static/doctors.csv', 'r') as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             doctors = [row for row in csv_reader if row['Specialization'] == specialization]
#             return JsonResponse(doctors, safe=False)
#     else:
#         return JsonResponse({'error': 'Specialization not provided'})

# def doctors(request):
#     return render (request,"doctors.html")

# View to render the appointment booking form
def book_appointment(request):
    if request.method == 'POST':
        form = AppointmentForm(request.POST)
        if form.is_valid():
            # Process the form data
            form.save()
            # Add a success message
            messages.success(request, 'Your appointment was successfully booked!')
            # Redirect to the same page to show the message
            return redirect('book_appointment')
    else:
        form = AppointmentForm()
    
    return render(request, 'appointment.html', {'form': form})

# View to fetch doctor details based on specialization
def fetch_doctor_list(request):
    specialization = request.GET.get('specialization')
    csv_file_path = "E:/Medchoicehub/MyProject/static/doctors.csv"  # Update with your CSV file path
    doctors = []

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row['Specialization'].lower() == specialization.lower():
                    doctors.append(row)
    except FileNotFoundError:
        print("CSV file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return JsonResponse(doctors, safe=False)


def symptoms_checker(request):
    if request.method == 'POST':
        # Process the form data here, assuming you have logic to determine the disease
        # Replace 'Some Disease' with the actual disease determined from form data
        disease = 'Some Disease'
        return render(request, 'result.html', {'disease': disease})

    return render(request, 'symptoms_checker.html')