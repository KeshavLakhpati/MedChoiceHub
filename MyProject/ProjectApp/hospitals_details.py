import csv
from django.shortcuts import render

def hospital_details(request):
    csv_file_path = "E:\Medchoicehub\MyProject\static\data.csv"  # Update with your CSV file path
    hospitals = []

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            hospitals.append(row)

    return render(request, 'hospitals.html', {'hospitals': hospitals})

